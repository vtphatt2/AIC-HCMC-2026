from __future__ import annotations

import os
import threading
from pathlib import Path
from app.services.video_quarantine import excluded_video_ids

REMOTE_ROOT = Path(__file__).resolve().parents[2]


def artifact_path(env_name: str, default: str) -> Path:
    path = Path(os.getenv(env_name, default))
    return path if path.is_absolute() else REMOTE_ROOT / path


CAGRA_INDEX_PATH = artifact_path("CAGRA_INDEX_PATH", "cache/cagra/index.bin")
CAGRA_FRAME_IDS_PATH = artifact_path("CAGRA_FRAME_IDS_PATH", "cache/cagra/frame_ids.txt")
CAGRA_SEARCH_WIDTH = int(os.getenv("CAGRA_SEARCH_WIDTH", "32"))
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1280"))


class CagraClient:
    def __init__(self) -> None:
        try:
            import cupy as cp
            from cuvs.neighbors import cagra
        except ImportError as exc:
            raise RuntimeError(
                "CAGRA dependencies are missing. Install requirements-cagra.txt."
            ) from exc

        if not CAGRA_INDEX_PATH.is_file() or not CAGRA_FRAME_IDS_PATH.is_file():
            raise FileNotFoundError(
                "CAGRA index artifacts are missing. Run "
                "python scripts/build_cagra_index.py first."
            )

        self._cp = cp
        self._cagra = cagra
        self._index = cagra.load(str(CAGRA_INDEX_PATH))
        self._frame_ids = CAGRA_FRAME_IDS_PATH.read_text(encoding="utf-8").splitlines()
        self._lock = threading.Lock()

        if not self._frame_ids:
            raise RuntimeError(f"CAGRA frame ID file is empty: {CAGRA_FRAME_IDS_PATH}")
        if self._index.dataset.shape[0] != len(self._frame_ids):
            raise RuntimeError("CAGRA index and frame ID count do not match")
        if self._index.dim != VECTOR_DIM:
            raise RuntimeError(
                f"CAGRA index dim is {self._index.dim}; expected {VECTOR_DIM}"
            )

        import numpy as np

        self.search(np.ones(self._index.dim, dtype="float32"), top_k=100)

    def search(self, query_vector, top_k: int = 100) -> list[dict]:
        import numpy as np

        top_k = max(1, min(int(top_k), len(self._frame_ids)))
        wanted = top_k
        blocked = excluded_video_ids()
        if blocked:
            if getattr(self, '_quarantine_ids', None) != blocked:
                self._quarantine_count = sum(
                    frame.rsplit('_', 1)[0] in blocked for frame in self._frame_ids
                )
                self._quarantine_ids = blocked
            top_k = min(len(self._frame_ids), top_k + self._quarantine_count)
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise ValueError("CAGRA query vector has zero norm")
        query /= norm
        itopk_size = 1 << max(6, (top_k - 1).bit_length())
        params = self._cagra.SearchParams(
            itopk_size=itopk_size,
            search_width=CAGRA_SEARCH_WIDTH,
            algo="multi_cta",
        )

        with self._lock:
            distances, neighbors = self._cagra.search(
                params,
                self._index,
                self._cp.asarray(query),
                top_k,
            )
            self._cp.cuda.Stream.null.synchronize()
            scores = self._cp.asnumpy(self._cp.asarray(distances))[0]
            indices = self._cp.asnumpy(self._cp.asarray(neighbors))[0]

        hits = []
        for index, score in zip(indices, scores):
            frame_id = self._frame_ids[int(index)]
            video_id, frame_stem = frame_id.rsplit("_", 1)
            if video_id in blocked:
                continue
            hits.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "frame_number": int(frame_stem),
                    "timestamp_ms": 0,
                    "image_url": f"/static/frames/{video_id}/{frame_stem}.jpg",
                    "score": float(score),
                    "_timestamp_from_fps": True,
                }
            )
        return hits[:wanted]
