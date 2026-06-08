import os
import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

ENV_MODE = os.getenv("ENV_MODE", "MOCK")
REMOTE_SERVER_URL = os.getenv("REMOTE_SERVER_URL", "").rstrip("/")

MOCK_DIR = Path(__file__).parent / "mock"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_ROOT = next(
    (
        path
        for path in (REPO_ROOT / "AIC2026_sample", REPO_ROOT.parent / "AIC2026_sample")
        if path.is_dir()
    ),
    REPO_ROOT / "AIC2026_sample",
)
SAMPLE_ROOT = Path(os.getenv("AIC_SAMPLE_ROOT", DEFAULT_SAMPLE_ROOT))
logger = logging.getLogger(__name__)


def sample_subdir(name: str) -> Path:
    outer = SAMPLE_ROOT / name
    nested = outer / name
    if nested.is_dir():
        return nested
    return outer


class DataProvider:
    """
    Switches data source based on ENV_MODE:

      MOCK  — returns data from app/mock/*.json (no server needed)
      SAMPLE — reads AIC2026_sample keyframes/metadata directly on this machine
      LOCAL — proxies requests to the GPU server at REMOTE_SERVER_URL
    """

    def __init__(self):
        self.mode = ENV_MODE
        self._feature_matrix = None
        self._feature_frame_ids = []
        self._frames_by_id = {}
        self._text_encoder = None

        if self.mode == "LOCAL":
            if not REMOTE_SERVER_URL:
                raise RuntimeError(
                    "ENV_MODE=LOCAL requires REMOTE_SERVER_URL to be set in .env\n"
                    "Example: REMOTE_SERVER_URL=https://xxxx.ngrok.io"
                )
            print(f"DataProvider: LOCAL mode → {REMOTE_SERVER_URL}")

        elif self.mode == "MOCK":
            self._videos, self._frames, self._ocr, self._transcripts = self._load_mock()
            print(
                f"DataProvider: MOCK mode — "
                f"{len(self._videos)} videos, {len(self._frames)} frames loaded"
            )

        elif self.mode == "SAMPLE":
            self._videos, self._frames = self._load_sample()
            self._frames_by_id = {frame["frame_id"]: frame for frame in self._frames}
            self._ocr = []
            self._transcripts = []
            print(
                f"DataProvider: SAMPLE mode — "
                f"{len(self._videos)} videos, {len(self._frames)} keyframes loaded"
            )

        else:
            raise RuntimeError(
                f"Unknown ENV_MODE='{self.mode}'. Valid values: MOCK, SAMPLE, LOCAL"
            )

    # ── Public interface ───────────────────────────────────────────────────────

    async def get_raw_data(self, query_groups: list[dict], limit: int = 1000) -> dict:
        """
        Returns a unified dict consumed by strategy.fusion_and_temporal():
          frames, ocr, transcripts (lists), videos (dict keyed by video_id)
        """
        if self.mode == "MOCK":
            return self._local_raw_data(limit)
        if self.mode == "SAMPLE":
            return self._sample_raw_data(query_groups, limit)
        return await self._fetch_from_remote(query_groups, limit)

    # ── MOCK ──────────────────────────────────────────────────────────────────

    def _load_mock(self):
        def load(name):
            return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))

        videos_list = load("mock_videos.json")
        frames = load("mock_frames.json")
        ocr = load("mock_ocr.json")
        transcripts = load("mock_transcripts.json")
        return videos_list, frames, ocr, transcripts

    def _local_raw_data(self, limit: int) -> dict:
        videos_by_id = {v["video_id"]: v for v in self._videos}
        return {
            "frames":      self._frames[:limit],
            "ocr":         self._ocr[:limit],
            "transcripts": self._transcripts[:limit],
            "videos":      videos_by_id,
        }

    # ── SAMPLE ───────────────────────────────────────────────────────────────

    def _load_sample(self):
        metadata_dir = sample_subdir("metadata")
        keyframes_dir = sample_subdir("keyframes")
        features_dir = sample_subdir("PECore-features")

        for label, path in {
            "metadata": metadata_dir,
            "keyframes": keyframes_dir,
            "PECore-features": features_dir,
        }.items():
            if not path.is_dir():
                raise RuntimeError(f"AIC sample {label} directory not found: {path}")

        videos = []
        frames = []

        for metadata_path in sorted(metadata_dir.glob("*.json")):
            video_id = metadata_path.stem
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            fps = float(data.get("fps") or 25.0)
            frame_paths = sorted((features_dir / video_id).glob("*.npy"))
            if not frame_paths:
                frame_paths = sorted((keyframes_dir / video_id).glob("*.jpg"))

            youtube_id = data.get("youtube_id") or self._youtube_id_from_link(
                data.get("video_link", ""),
                fallback=video_id,
            )
            max_timestamp_ms = 0

            for path in frame_paths:
                frame_number = int(path.stem)
                timestamp_ms = int(frame_number / fps * 1000)
                max_timestamp_ms = max(max_timestamp_ms, timestamp_ms)
                frames.append({
                    "frame_id":     f"{video_id}_{path.stem}",
                    "video_id":     video_id,
                    "frame_number": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "image_url":    f"/static/frames/{video_id}/{path.stem}.jpg",
                    "_feature_path": str(path) if path.suffix == ".npy" else "",
                })

            videos.append({
                "video_id":    video_id,
                "title":       data.get("title") or video_id,
                "youtube_id":  youtube_id,
                "fps":         fps,
                "duration_ms": max_timestamp_ms,
                "frame_count": len(frame_paths),
            })

        frames.sort(key=lambda f: (f["video_id"], f["frame_number"]))
        return videos, frames

    # ── SAMPLE linear vector search ──────────────────────────────────────────

    def _sample_raw_data(self, query_groups: list[dict], limit: int) -> dict:
        frames = []
        for group_index, group in enumerate(query_groups):
            semantic_query = str(group.get("semantic_query", "")).strip()
            if not semantic_query:
                continue

            query_vector = self._encode_sample_text(semantic_query)
            hits = self.linear_search_by_vector(query_vector, top_k=limit)
            for hit in hits:
                hit["_query_group_index"] = group_index
            frames.extend(hits)
            logger.info("SAMPLE semantic query group=%s returned %s hits", group_index, len(hits))

        if not frames:
            frames = self._frames[:limit]

        videos_by_id = {v["video_id"]: v for v in self._videos}
        return {
            "frames":      self._dedupe_sample_frames(frames),
            "ocr":         self._ocr[:limit],
            "transcripts": self._transcripts[:limit],
            "videos":      videos_by_id,
        }

    def _encode_sample_text(self, text: str):
        if self._text_encoder is None:
            from app.services.text_encoder import PECoreTextEncoder

            self._text_encoder = PECoreTextEncoder()
        return self._text_encoder.encode(text)

    def resolve_sample_frame_id(self, query: str) -> str | None:
        """Accept L01_V001_000022, L01_V001/000022, or L01_V001 000022."""
        if self.mode != "SAMPLE":
            return None

        query = query.strip().replace("\\", "/")
        if not query:
            return None

        candidates = [query]
        if "/" in query:
            video_id, frame = query.rsplit("/", 1)
            candidates.append(f"{video_id}_{Path(frame).stem}")
        parts = query.split()
        if len(parts) == 2:
            candidates.append(f"{parts[0]}_{Path(parts[1]).stem}")

        for candidate in candidates:
            if candidate in self._frames_by_id:
                return candidate
        return None

    def linear_search_by_frame_id(self, frame_id: str, top_k: int = 100) -> list[dict]:
        if self.mode != "SAMPLE":
            raise RuntimeError("linear_search_by_frame_id is only available in ENV_MODE=SAMPLE")
        self._ensure_feature_index()

        if frame_id not in self._feature_frame_ids:
            raise ValueError(f"Frame '{frame_id}' has no PECore feature vector")

        query_idx = self._feature_frame_ids.index(frame_id)
        query_vector = self._feature_matrix[query_idx]
        return self.linear_search_by_vector(query_vector, top_k=top_k)

    def linear_search_by_vector(self, query_vector, top_k: int = 100) -> list[dict]:
        if self.mode != "SAMPLE":
            raise RuntimeError("linear_search_by_vector is only available in ENV_MODE=SAMPLE")
        self._ensure_feature_index()

        import numpy as np

        query_vector = np.asarray(query_vector, dtype="float32").reshape(-1)
        if query_vector.shape[0] != self._feature_matrix.shape[1]:
            raise ValueError(
                f"Query vector has dim {query_vector.shape[0]}; "
                f"expected {self._feature_matrix.shape[1]}"
            )
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Query vector has zero norm")

        query_vector = query_vector / norm
        scores = self._feature_matrix @ query_vector
        top_k = max(1, min(int(top_k), len(scores)))
        partition_k = min(top_k - 1, len(scores) - 1)
        top_indices = np.argpartition(-scores, partition_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            hit_frame_id = self._feature_frame_ids[int(idx)]
            frame = dict(self._frames_by_id[hit_frame_id])
            frame["score"] = float(scores[int(idx)])
            results.append(frame)
        return results

    def _ensure_feature_index(self) -> None:
        if self._feature_matrix is not None:
            return

        import numpy as np

        vectors = []
        frame_ids = []
        for frame in self._frames:
            feature_path = frame.get("_feature_path")
            if not feature_path:
                continue
            vector = np.load(feature_path).astype("float32").reshape(-1)
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            vectors.append(vector / norm)
            frame_ids.append(frame["frame_id"])

        if not vectors:
            raise RuntimeError("No PECore .npy feature vectors found for SAMPLE mode")

        self._feature_matrix = np.vstack(vectors)
        self._feature_frame_ids = frame_ids
        print(
            f"DataProvider: built linear PECore index — "
            f"{self._feature_matrix.shape[0]} vectors x {self._feature_matrix.shape[1]} dims"
        )

    @staticmethod
    def _dedupe_sample_frames(frames: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for frame in frames:
            identity = (frame.get("frame_id"), frame.get("_query_group_index"))
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(frame)
        return deduped

    @staticmethod
    def _youtube_id_from_link(link: str, fallback: str) -> str:
        parsed = urlparse(link)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/") or fallback
        return parse_qs(parsed.query).get("v", [fallback])[0] or fallback

    # ── LOCAL (proxy to remote server) ────────────────────────────────────────

    async def _fetch_from_remote(self, query_groups: list[dict], limit: int) -> dict:
        async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=15.0) as client:
            response = await client.post(
                "/api/raw-data",
                json={"query_groups": query_groups, "limit": limit},
            )
            response.raise_for_status()
            return response.json()
