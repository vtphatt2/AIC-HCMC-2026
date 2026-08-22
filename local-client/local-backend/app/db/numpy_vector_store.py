"""Exact vector search over a memory-mapped array.

Local search has to be exact regardless of speed: milvus_lite 3.2.0's HNSW path
returns wrong neighbours (docs/archive/milvus-lite-hnsw-recall-bug.md). Given that, the
only question is which exact search to run. Measured over the same 990 MB,
top_k=1000:

    milvus_lite BruteForceIndex   ~1570 ms
    this module                     ~35 ms

The gap is not cleverness — it is BLAS. One `block @ query` is a threaded GEMV
running at tens of GB/s, where the Milvus path adds per-row Python and gRPC
serialisation on top of the same arithmetic.

Built by scripts/export_vectors_npy.py, which reads the ingest archives
directly. Rerun it whenever you re-ingest.

Scanning in chunks is not an optimisation — chunk size barely moves the number
(34/35/37 ms at 16k/64k/193k rows). It is so the same code holds when the
corpus outgrows RAM: peak extra memory is one block of scores, and the OS pages
the rest through the memmap.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_ROWS = int(os.getenv("NUMPY_SEARCH_CHUNK_ROWS", "65536"))

_lock = threading.Lock()
_store: "_VectorStore | None" = None


class _VectorStore:
    def __init__(self, vectors_path: Path, meta_path: Path):
        # mmap_mode leaves the 990 MB on disk and lets the OS page it in. It
        # also means the pages are evictable under pressure instead of
        # committed, which matters on a machine already holding Milvus.
        self.vectors = np.load(vectors_path, mmap_mode="r")
        meta = np.load(meta_path, allow_pickle=False)
        self.frame_id = meta["frame_id"]
        self.video_id = meta["video_id"]
        self.video_genre = meta["video_genre"]
        self.frame_number = meta["frame_number"]
        self.timestamp_ms = meta["timestamp_ms"]
        self.youtube_id = meta["youtube_id"]

        if self.vectors.shape[0] != self.frame_id.shape[0]:
            raise ValueError(
                f"{vectors_path.name} has {self.vectors.shape[0]} rows but "
                f"{meta_path.name} has {self.frame_id.shape[0]} — re-run "
                f"scripts/export_vectors_npy.py"
            )
        self._row_of = {str(f): i for i, f in enumerate(self.frame_id)}
        logger.info(
            "numpy vector store: %s vectors dim %s from %s",
            self.vectors.shape[0], self.vectors.shape[1], vectors_path.parent,
        )

    def rows_for(self, frame_ids) -> np.ndarray:
        found = [self._row_of[f] for f in map(str, frame_ids) if f in self._row_of]
        return np.asarray(found, dtype="int64")


def paths() -> tuple[Path, Path] | None:
    """Where export_vectors_npy.py writes, next to the Milvus Lite database."""
    db = os.getenv("MILVUS_LITE_PATH", "").strip()
    if not db:
        return None
    base = Path(db).parent
    vectors, meta = base / "vectors.f32.npy", base / "vectors.meta.npz"
    return (vectors, meta) if vectors.is_file() and meta.is_file() else None


def available() -> bool:
    return paths() is not None


def staleness_warning() -> str | None:
    """Ingesting without re-exporting leaves search running on the previous
    vectors, and nothing else would notice — the file is still valid, just
    older than the archives it came from. Compare timestamps and say so."""
    found = paths()
    if found is None:
        return None
    vectors, _ = found
    zip_dir = vectors.parent / "zip_embeddings"
    if not zip_dir.is_dir():
        return None
    newest = max((p.stat().st_mtime for p in zip_dir.glob("*_results.zip")), default=None)
    if newest is None or newest <= vectors.stat().st_mtime:
        return None
    return (
        f"{vectors.name} is older than the archives in {zip_dir.name}/. "
        f"Search is running on vectors from before the last ingest. "
        f"Re-run: python scripts/export_vectors_npy.py"
    )


def get_store() -> _VectorStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                found = paths()
                if found is None:
                    raise RuntimeError(
                        "No vectors.f32.npy found — run "
                        "scripts/export_vectors_npy.py"
                    )
                _store = _VectorStore(*found)
    return _store


def frame_vectors(frame_ids: list[str]) -> dict[str, list[float]]:
    """Raw vectors by frame_id, for the near-duplicate result filter. Free
    compared with the Milvus path — the rows are already mapped, so this is a
    copy rather than 1280 floats per hit over gRPC."""
    store = get_store()
    rows = store.rows_for(frame_ids)
    return {
        str(store.frame_id[row]): np.asarray(store.vectors[row], dtype="float32").tolist()
        for row in rows.tolist()
    }


def frames_in_range(
    video_id: str, start_ms: int, end_ms: int, *, limit: int = 20
) -> list[dict]:
    """Indexed keyframes of one video inside a time window, in time order."""
    store = get_store()
    selected = (
        (store.video_id == video_id)
        & (store.timestamp_ms >= start_ms)
        & (store.timestamp_ms <= end_ms)
    )
    rows = np.flatnonzero(selected)
    rows = rows[np.argsort(store.timestamp_ms[rows], kind="stable")][: max(1, int(limit))]
    return [
        {
            "frame_id": str(store.frame_id[row]),
            "video_id": str(store.video_id[row]),
            "frame_number": int(store.frame_number[row]),
            "timestamp_ms": int(store.timestamp_ms[row]),
            "image_url": "",
            "youtube_id": str(store.youtube_id[row]),
        }
        for row in rows.tolist()
    ]


def context_frames(
    video_id: str, start_ms: int, end_ms: int, *, expand: int = 20
) -> tuple[list[dict], list[dict]]:
    """Up to `expand` indexed keyframes immediately before start_ms and up to
    `expand` immediately after end_ms — for the Video view strip's "expand
    a sparse result cluster with neighboring context frames" feature. Not a
    fixed time window (keyframe spacing varies a lot — see
    docs/archive/keyframe_selection.md), a fixed *count* either side."""
    store = get_store()
    rows = np.flatnonzero(store.video_id == video_id)
    rows = rows[np.argsort(store.timestamp_ms[rows], kind="stable")]
    ts = store.timestamp_ms[rows]
    lo = int(np.searchsorted(ts, start_ms, side="left"))
    hi = int(np.searchsorted(ts, end_ms, side="right"))
    expand = max(0, int(expand))

    def to_dicts(selected_rows) -> list[dict]:
        return [
            {
                "frame_id": str(store.frame_id[row]),
                "video_id": str(store.video_id[row]),
                "frame_number": int(store.frame_number[row]),
                "timestamp_ms": int(store.timestamp_ms[row]),
                "image_url": "",
                "youtube_id": str(store.youtube_id[row]),
            }
            for row in selected_rows.tolist()
        ]

    return to_dicts(rows[max(0, lo - expand):lo]), to_dicts(rows[hi:hi + expand])


def vector_search(
    query_vector: list[float],
    top_k: int = 100,
    *,
    exclude_frame_ids: list[str] | None = None,
    video_genre: str = "All",
    include_vector: bool = False,
) -> list[dict]:
    """Exact top-k by cosine. Mirrors milvus_client.vector_search's output so
    callers cannot tell which backend answered."""
    store = get_store()
    top_k = max(1, int(top_k))

    q = np.asarray(query_vector, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("query vector has zero norm")
    q /= norm

    n = store.vectors.shape[0]
    blocked = np.zeros(n, dtype=bool)
    if exclude_frame_ids:
        blocked[store.rows_for(exclude_frame_ids)] = True
    if video_genre and video_genre != "All":
        blocked |= store.video_genre != video_genre
    if blocked.all():
        return []

    best_scores = np.empty(0, dtype="float32")
    best_rows = np.empty(0, dtype="int64")
    for start in range(0, n, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n)
        scores = np.asarray(store.vectors[start:stop]) @ q
        # -inf rather than dropping rows, so positions stay aligned with the
        # block and the partition below stays a single pass.
        scores[blocked[start:stop]] = -np.inf
        k = min(top_k, scores.shape[0])
        part = np.argpartition(-scores, k - 1)[:k]
        best_scores = np.concatenate([best_scores, scores[part]])
        best_rows = np.concatenate([best_rows, part + start])
        if best_scores.shape[0] > top_k * 4:
            keep = np.argpartition(-best_scores, top_k - 1)[:top_k]
            best_scores, best_rows = best_scores[keep], best_rows[keep]

    order = np.argsort(-best_scores)[:top_k]
    hits = []
    for pos in order:
        score = float(best_scores[pos])
        if score == -np.inf:
            break
        row = int(best_rows[pos])
        hit = {
            "frame_id": str(store.frame_id[row]),
            "video_id": str(store.video_id[row]),
            "frame_number": int(store.frame_number[row]),
            "timestamp_ms": int(store.timestamp_ms[row]),
            "image_url": "",
            "youtube_id": str(store.youtube_id[row]),
            "score": score,
        }
        if include_vector:
            # Free here, unlike the Milvus path where it is 1280 floats per hit
            # over gRPC — the row is already mapped.
            hit["_vector"] = np.asarray(store.vectors[row], dtype="float32").tolist()
        hits.append(hit)
    return hits
