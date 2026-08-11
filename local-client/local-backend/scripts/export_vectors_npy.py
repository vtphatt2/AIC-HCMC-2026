#!/usr/bin/env python3
"""Export the keyframe embeddings to a flat .npy the backend can memory-map.

Reads the same `*_results.zip` archives the ingest reads, not Milvus. The
vectors are already sitting in there as `phase2_embeddings/video__<id>/
embeddings.npy`, so going through Milvus to get them back would mean paging
193k rows out over gRPC one video at a time — minutes of work to recover bytes
we can read directly in seconds, and it needs the backend stopped for the
single-process lock.

Why a flat array at all: milvus_lite 3.2.0's HNSW search returns wrong
neighbours (docs/milvus-lite-hnsw-recall-bug.md), so local search has to be
exact either way. Its BruteForceIndex manages roughly 1 GB/s over the same
bytes; a memmapped array through BLAS should do several times that.

Run after every ingest, alongside it:

    python scripts/export_vectors_npy.py

Writes next to the Milvus Lite database, row-aligned with each other:

    vectors.f32.npy    (N, 1280) float32, C-contiguous, L2-normalised
    vectors.meta.npz   frame_id / video_id / video_genre / frame_number /
                       timestamp_ms / youtube_id
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "remote-server"))

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

# Reuse the ingest's archive conventions rather than restating them. Only the
# per-row assembly is rewritten here, because iter_video_records() calls
# .tolist() on every vector — fine for a gRPC upsert, but 193k x 1280 Python
# floats is several GB when the destination is a numpy array anyway.
from scripts.ingest_zip_pipeline_results import (  # noqa: E402
    VIDEO_DIR_PATTERN,
    iter_result_archives,
    load_media_info,
)

META_FIELDS = ["frame_id", "video_id", "video_genre", "frame_number", "timestamp_ms", "youtube_id"]


def export(zip_dir: Path, out_dir: Path) -> int:
    media_info = load_media_info(zip_dir)
    archives = iter_result_archives(zip_dir)
    print(f"reading {len(archives)} archives from {zip_dir}")

    t0 = time.time()
    blocks: list[np.ndarray] = []
    meta: dict[str, list] = {f: [] for f in META_FIELDS}

    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            video_ids = sorted({
                match.group("video_id")
                for name in names
                if name.startswith("phase1_transnet/")
                for match in [VIDEO_DIR_PATTERN.match(name.split("/", 2)[1])]
                if match
            })
            before = sum(len(b) for b in blocks)
            for video_id in video_ids:
                embeddings_path = f"phase2_embeddings/video__{video_id}/embeddings.npy"
                if embeddings_path not in names:
                    print(f"  {video_id}: no embeddings.npy, skipped")
                    continue

                scenes = json.loads(zf.read(f"phase1_transnet/video__{video_id}/scenes.json"))
                keyframes = json.loads(zf.read(f"phase1_transnet/video__{video_id}/keyframes.json"))
                fps = float(scenes["fps"])
                selected = keyframes["keyframes"]

                vectors = np.load(io.BytesIO(zf.read(embeddings_path))).astype("float32")
                if vectors.shape[0] != len(selected):
                    raise ValueError(
                        f"{video_id}: {vectors.shape[0]} embedding rows but "
                        f"{len(selected)} keyframes (row order must match 1:1)"
                    )

                youtube_id = media_info.get(video_id, {}).get("youtube_id", "")
                frame_numbers = np.asarray([int(k["frame_number"]) for k in selected], dtype="int64")

                blocks.append(vectors)
                meta["frame_id"].extend(f"{video_id}_{n:06d}" for n in frame_numbers)
                meta["video_id"].extend([video_id] * len(selected))
                meta["video_genre"].extend([""] * len(selected))
                meta["frame_number"].extend(frame_numbers.tolist())
                meta["timestamp_ms"].extend((frame_numbers / fps * 1000).astype("int64").tolist())
                meta["youtube_id"].extend([youtube_id] * len(selected))

            added = sum(len(b) for b in blocks) - before
            print(f"  {archive.name:<26} {len(video_ids):>4} videos, {added:>7} keyframes "
                  f"({time.time() - t0:.0f}s)")

    mat = np.vstack(blocks)
    # The ingest normalises on the way in; do it here too so search is a plain
    # dot product and this file is correct on its own terms rather than
    # inheriting an assumption.
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = np.ascontiguousarray(mat / norms, dtype="float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "vectors.f32.npy"
    meta_path = out_dir / "vectors.meta.npz"
    try:
        np.save(vec_path, mat)
    except OSError as exc:
        # A running backend has this file memory-mapped, and Windows will not
        # let a mapped file be overwritten or renamed — so writing to a temp
        # name and swapping does not help either. Same shape as the Milvus Lite
        # single-process lock: stop the backend first.
        raise SystemExit(
            f"Could not write {vec_path.name}: {exc}\n"
            f"A running local-backend keeps it memory-mapped. Stop the backend, "
            f"re-run this, then start it again."
        ) from exc
    np.savez(
        meta_path,
        **{f: np.asarray(meta[f], dtype="int64" if f in ("frame_number", "timestamp_ms") else None)
           for f in META_FIELDS},
    )

    print(f"\n{vec_path.name}  {mat.shape}  {vec_path.stat().st_size / 1e9:.2f} GB")
    print(f"{meta_path.name}  {len(meta['frame_id'])} rows  "
          f"{meta_path.stat().st_size / 1e6:.0f} MB")
    print(f"total {time.time() - t0:.0f}s")
    return 0


def main() -> int:
    default_zip = REPO_ROOT / "challenge_resources" / "data" / "zip_file"
    db_path = os.getenv("MILVUS_LITE_PATH", "").strip()
    default_out = Path(db_path).parent if db_path else REPO_ROOT / "challenge_resources" / "data"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-dir", type=Path, default=default_zip)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    args = parser.parse_args()
    return export(args.zip_dir, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
