"""
Compare local brute-force PE-Core search against Milvus HNSW search.

Run from remote-server:
  python scripts/compare_linear_vs_milvus.py --query "a busy street with people"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REMOTE_ROOT.parent
sys.path.insert(0, str(REMOTE_ROOT))

from scripts.sample_paths import default_sample_root, sample_subdir

logger = logging.getLogger("compare_linear_vs_milvus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare linear PE-Core search with Milvus search.")
    parser.add_argument("--query", required=True, help="Semantic text query.")
    parser.add_argument("--sample-root", type=Path, default=default_sample_root(REPO_ROOT))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--linear-pool", type=int, default=100)
    parser.add_argument(
        "--include-cagra",
        action="store_true",
        help="Also compare the prepared CAGRA index.",
    )
    return parser.parse_args()


def iter_feature_paths(sample_root: Path) -> list[tuple[str, Path]]:
    features_dir = sample_subdir(sample_root, "PECore-features")
    if not features_dir.is_dir():
        raise FileNotFoundError(f"PE-Core features directory not found: {features_dir}")

    paths: list[tuple[str, Path]] = []
    for video_dir in sorted(path for path in features_dir.iterdir() if path.is_dir()):
        if video_dir.name == "selected_keyframes":
            continue
        for feature_path in sorted(video_dir.glob("*.npy")):
            paths.append((f"{video_dir.name}_{feature_path.stem}", feature_path))
    return paths


def linear_search(query_vector: Any, sample_root: Path, top_k: int) -> list[dict[str, Any]]:
    import numpy as np

    feature_paths = iter_feature_paths(sample_root)
    vectors = []
    frame_ids = []
    for frame_id, path in tqdm(feature_paths, desc="Loading linear-search vectors", unit="vector"):
        vector = np.load(path).astype("float32").reshape(-1)
        norm = np.linalg.norm(vector)
        if norm == 0:
            logger.warning("Skipping zero-norm vector: %s", path)
            continue
        vectors.append(vector / norm)
        frame_ids.append(frame_id)

    if not vectors:
        raise RuntimeError(f"No PE-Core vectors found under {sample_root}")

    matrix = np.vstack(vectors)
    query_vector = np.asarray(query_vector, dtype="float32").reshape(-1)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Query vector has zero norm")
    query_vector = query_vector / query_norm

    scores = matrix @ query_vector
    top_k = max(1, min(top_k, len(scores)))
    top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [
        {"frame_id": frame_ids[int(idx)], "score": float(scores[int(idx)])}
        for idx in top_indices
    ]


def overlap_at(linear_hits: list[dict[str, Any]], milvus_hits: list[dict[str, Any]], k: int) -> int:
    linear_ids = {row["frame_id"] for row in linear_hits[:k]}
    milvus_ids = {row["frame_id"] for row in milvus_hits[:k]}
    return len(linear_ids & milvus_ids)


def score_diff(linear_hits: list[dict[str, Any]], milvus_hits: list[dict[str, Any]]) -> dict[str, float]:
    linear_scores = {row["frame_id"]: float(row["score"]) for row in linear_hits}
    milvus_scores = {row["frame_id"]: float(row["score"]) for row in milvus_hits}
    shared_ids = sorted(set(linear_scores) & set(milvus_scores))
    if not shared_ids:
        return {"shared": 0.0, "mean_abs": 0.0, "max_abs": 0.0}

    diffs = [abs(linear_scores[frame_id] - milvus_scores[frame_id]) for frame_id in shared_ids]
    return {
        "shared": float(len(shared_ids)),
        "mean_abs": sum(diffs) / len(diffs),
        "max_abs": max(diffs),
    }


def print_comparison(
    linear_hits: list[dict[str, Any]],
    backend_hits: list[dict[str, Any]],
    top_k: int,
    backend_name: str,
) -> None:
    diff = score_diff(linear_hits, backend_hits)
    print(f"overlap@5:  {overlap_at(linear_hits, backend_hits, 5)}/5")
    print(f"overlap@10: {overlap_at(linear_hits, backend_hits, 10)}/10")
    print(
        "score_difference_on_shared: "
        f"shared={int(diff['shared'])}, mean_abs={diff['mean_abs']:.6f}, max_abs={diff['max_abs']:.6f}"
    )
    print(f"\nrank  linear_frame_id      linear_score  {backend_name}_frame_id      {backend_name}_score")
    print("----  -------------------  ------------  -------------------  ------------")
    for rank in range(top_k):
        linear = linear_hits[rank] if rank < len(linear_hits) else {}
        backend = backend_hits[rank] if rank < len(backend_hits) else {}
        print(
            f"{rank + 1:<4}  "
            f"{str(linear.get('frame_id', '')):<19}  "
            f"{float(linear.get('score') or 0.0):<12.6f}  "
            f"{str(backend.get('frame_id', '')):<19}  "
            f"{float(backend.get('score') or 0.0):<12.6f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    from app.db import milvus_client
    from app.services.text_encoder import PECoreTextEncoder

    encoder = PECoreTextEncoder()
    query_vector = encoder.encode(args.query)
    linear_top = max(args.top_k, args.linear_pool)

    logger.info("Running linear search over %s", args.sample_root)
    linear_hits = linear_search(query_vector, args.sample_root, linear_top)

    milvus_client.connect()
    collection = milvus_client.get_collection()
    logger.info("Running Milvus search collection=%s", collection.name)
    milvus_hits = milvus_client.vector_search(collection, query_vector.tolist(), top_k=linear_top)

    print(f"query: {args.query}")
    print("\nbackend: milvus")
    print_comparison(linear_hits, milvus_hits, args.top_k, "milvus")

    if args.include_cagra:
        from app.db.cagra_client import CagraClient

        logger.info("Running CAGRA search")
        cagra_hits = CagraClient().search(query_vector, top_k=linear_top)
        print("\nbackend: cagra")
        print_comparison(linear_hits, cagra_hits, args.top_k, "cagra")


if __name__ == "__main__":
    main()
