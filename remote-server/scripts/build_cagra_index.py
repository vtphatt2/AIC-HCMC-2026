"""Build a persisted cuVS CAGRA index from AIC2026_sample PE-Core vectors."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REMOTE_ROOT.parent
sys.path.insert(0, str(REMOTE_ROOT))

from scripts.sample_paths import default_sample_root, sample_subdir

logger = logging.getLogger("build_cagra_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cuVS CAGRA index.")
    parser.add_argument("--sample-root", type=Path, default=default_sample_root(REPO_ROOT))
    parser.add_argument("--output-dir", type=Path, default=REMOTE_ROOT / "cache" / "cagra")
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--intermediate-graph-degree", type=int, default=128)
    return parser.parse_args()


def feature_paths(sample_root: Path) -> list[tuple[str, Path]]:
    root = sample_subdir(sample_root, "PECore-features")
    paths = []
    for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if video_dir.name == "selected_keyframes":
            continue
        paths.extend(
            (f"{video_dir.name}_{path.stem}", path)
            for path in sorted(video_dir.glob("*.npy"))
        )
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    paths = feature_paths(args.sample_root.resolve())
    if not paths:
        raise RuntimeError(f"No PE-Core vectors found under {args.sample_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = args.output_dir / "vectors.npy"
    frame_ids_path = args.output_dir / "frame_ids.txt"
    index_path = args.output_dir / "index.bin"

    first = np.load(paths[0][1]).astype("float32").reshape(-1)
    vectors = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype="float32",
        shape=(len(paths), first.size),
    )
    with frame_ids_path.open("w", encoding="utf-8") as frame_ids:
        for index, (frame_id, path) in enumerate(paths):
            vector = np.load(path).astype("float32").reshape(-1)
            norm = float(np.linalg.norm(vector))
            if vector.size != first.size or norm == 0.0:
                raise ValueError(f"Invalid PE-Core vector: {path}")
            vectors[index] = vector / norm
            frame_ids.write(f"{frame_id}\n")
    vectors.flush()

    import cupy as cp
    import cuvs
    from cuvs.neighbors import cagra

    started = time.perf_counter()
    index = cagra.build(
        cagra.IndexParams(
            metric="inner_product",
            graph_degree=args.graph_degree,
            intermediate_graph_degree=args.intermediate_graph_degree,
            build_algo="nn_descent",
        ),
        cp.asarray(vectors),
    )
    cp.cuda.Stream.null.synchronize()
    build_ms = (time.perf_counter() - started) * 1000
    cagra.save(str(index_path), index, include_dataset=True)
    del index
    del vectors
    vectors_path.unlink()

    manifest = {
        "cuvs_version": cuvs.__version__,
        "count": len(paths),
        "dim": int(first.size),
        "metric": "inner_product",
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
        "build_ms": build_ms,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Built CAGRA index: %s", manifest)


if __name__ == "__main__":
    main()
