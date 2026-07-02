"""
Benchmark vector-search algorithms against PECore text-query ground truth.

This script is intentionally benchmark-only. It does not integrate new search
algorithms into the app. It measures retrieval quality, latency, and memory
snapshots for each algorithm so results are comparable across runs.

Run from remote-server:
  python scripts/benchmark_vector_search_algorithms.py \
    --queries-json test_queries.json \
    --output-json benchmark_vector_search_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
REMOTE_ROOT = REPO_ROOT / "remote-server"
sys.path.insert(0, str(REMOTE_ROOT))

from app.db import milvus_client
from app.services.text_encoder import PECoreTextEncoder


BENCH_IVF_COLLECTION = "video_frames_ivf_flat_bench"
BENCH_SCANN_COLLECTION = "video_frames_scann_bench"
BENCH_HNSW_M16_COLLECTION = "video_frames_hnsw_m16_bench"
BENCH_HNSW_M32_COLLECTION = "video_frames_hnsw_m32_bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vector-search algorithms.")
    parser.add_argument("--queries-json", type=Path, default=REMOTE_ROOT / "test_queries.json")
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_vector_search_report.json"))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["flat", "hnsw", "hnsw_m32", "cagra", "gpu_bruteforce", "ivf_flat", "gpu_ivf_flat", "scann_raw"],
        help="Algorithms to benchmark.",
    )
    parser.add_argument("--ivf-nlist", type=int, default=0, help="0 means choose from dataset size.")
    parser.add_argument("--ivf-nprobe", type=int, default=0, help="0 means choose from dataset size.")
    parser.add_argument("--scann-reorder-k", type=int, default=0, help="0 means 5 * top_k.")
    parser.add_argument("--keep-temp-collections", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * p / 100)))]


def process_rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 2)
    except OSError:
        return None
    return None


def gpu_snapshot() -> dict[str, int] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    used_mb, total_mb, util_percent = [int(value.strip()) for value in output.splitlines()[0].split(",")]
    return {"used_mb": used_mb, "total_mb": total_mb, "util_percent": util_percent}


def docker_container_memory(container: str = "remote-server-milvus-1") -> dict[str, Any] | None:
    try:
        output = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    used = output.split("/")[0].strip()
    return {"container": container, "raw": output, "used": used}


def memory_snapshot(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "process_rss_mb": process_rss_mb(),
        "gpu": gpu_snapshot(),
        "milvus_container": docker_container_memory(),
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, query in enumerate(payload.get("queries", []), start=1):
        ground_truth = query.get("ground_truth") or {}
        target = query.get("target") or {}
        text = query.get("query_text") or query.get("query") or query.get("semantic_query")
        frame_ids = (
            target.get("ground_truth_frame_ids")
            or ground_truth.get("frame_ids")
            or query.get("ground_truth_frame_ids")
        )
        frame_id = (
            target.get("frame_id")
            or ground_truth.get("frame_id")
            or query.get("ground_truth_frame_id")
        )
        if frame_ids is None and frame_id:
            frame_ids = [frame_id]
        if not text or not frame_ids:
            continue
        rows.append(
            {
                "id": query.get("query_id") or query.get("id") or f"q{index:03d}",
                "text": text,
                "gt_frame_ids": set(frame_ids),
                "language": query.get("language") or query.get("benchmark", {}).get("language") or "unknown",
                "query_type": query.get("query_type") or query.get("benchmark", {}).get("query_type") or "visual_scene",
            }
        )
    if not rows:
        raise ValueError(f"No benchmark queries with text and ground truth found in {path}")
    return rows


def encode_queries(query_rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, float]]:
    encoder = PECoreTextEncoder()
    vectors = []
    latencies = []
    for row in query_rows:
        started = time.perf_counter()
        vector = np.asarray(encoder.encode(row["text"]), dtype=np.float32).reshape(-1)
        latencies.append((time.perf_counter() - started) * 1000)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        vectors.append(vector)
    return np.vstack(vectors).astype(np.float32), {
        "included_in_search_latency": False,
        "p50_ms": round(median(latencies), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "total_ms": round(sum(latencies), 3),
    }


def load_vectors_from_milvus(collection_name: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    collection = Collection(collection_name)
    rows: list[dict[str, Any]] = []
    vectors = []
    iterator = collection.query_iterator(
        batch_size=2048,
        expr="frame_number >= 0",
        output_fields=["frame_id", "video_id", "frame_number", "timestamp_ms", "image_url", "vector"],
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            for row in batch:
                vector = np.asarray(row.pop("vector"), dtype=np.float32).reshape(-1)
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                rows.append(dict(row))
                vectors.append(vector)
    finally:
        iterator.close()
    return rows, np.vstack(vectors).astype(np.float32)


def milvus_search(
    collection: Collection,
    query_vectors: np.ndarray,
    algorithm: str,
    top_k: int,
    nprobe: int | None = None,
    reorder_k: int | None = None,
) -> tuple[list[float], list[list[str]]]:
    if algorithm == "hnsw":
        params = {"ef": max(512, top_k)}
    elif algorithm == "flat":
        params = {}
    elif algorithm == "scann":
        params = {"nprobe": int(nprobe or 32), "reorder_k": max(top_k, int(reorder_k or top_k * 5))}
    else:
        params = {"nprobe": int(nprobe or 32)}

    latencies = []
    results = []
    for query_vector in query_vectors:
        started = time.perf_counter()
        response = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": milvus_client.METRIC_TYPE, "params": params},
            limit=top_k,
            output_fields=["frame_id"],
        )
        latencies.append((time.perf_counter() - started) * 1000)
        results.append([hit.entity.get("frame_id") for hit in response[0]])
    return latencies, results


def cagra_search(query_vectors: np.ndarray, top_k: int) -> tuple[list[float], list[list[str]]]:
    from app.db.cagra_client import CagraClient

    client = CagraClient()
    latencies = []
    results = []
    for query_vector in query_vectors:
        started = time.perf_counter()
        hits = client.search(query_vector, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        results.append([hit["frame_id"] for hit in hits])
    return latencies, results


def gpu_bruteforce_search(
    vectors: np.ndarray,
    frame_ids: list[str],
    query_vectors: np.ndarray,
    top_k: int,
) -> tuple[float, list[float], list[list[str]]]:
    import cupy as cp
    from cuvs.neighbors import brute_force

    dataset = cp.asarray(vectors)
    queries = cp.asarray(query_vectors)
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    index = brute_force.build(dataset, metric="inner_product")
    cp.cuda.Stream.null.synchronize()
    build_ms = (time.perf_counter() - started) * 1000

    latencies = []
    results = []
    for query_vector in queries:
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        _, neighbors = brute_force.search(index, query_vector.reshape(1, -1), top_k)
        cp.cuda.Stream.null.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        results.append([frame_ids[int(index)] for index in cp.asnumpy(neighbors)[0]])
    return build_ms, latencies, results


def gpu_ivf_search(
    vectors: np.ndarray,
    frame_ids: list[str],
    query_vectors: np.ndarray,
    top_k: int,
    n_lists: int,
    n_probes: int,
) -> tuple[float, list[float], list[list[str]]]:
    import cupy as cp
    from cuvs.neighbors import ivf_flat

    dataset = cp.asarray(vectors)
    queries = cp.asarray(query_vectors)
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    index = ivf_flat.build(
        ivf_flat.IndexParams(n_lists=n_lists, metric="inner_product", kmeans_n_iters=20),
        dataset,
    )
    cp.cuda.Stream.null.synchronize()
    build_ms = (time.perf_counter() - started) * 1000
    params = ivf_flat.SearchParams(n_probes=n_probes)

    latencies = []
    results = []
    for query_vector in queries:
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        _, neighbors = ivf_flat.search(params, index, query_vector.reshape(1, -1), top_k)
        cp.cuda.Stream.null.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        results.append([frame_ids[int(index)] for index in cp.asnumpy(neighbors)[0]])
    return build_ms, latencies, results


def create_benchmark_collection(
    collection_name: str,
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    index_type: str,
    index_params: dict[str, Any],
) -> Collection:
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    fields = [
        FieldSchema(name="frame_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="frame_number", dtype=DataType.INT64),
        FieldSchema(name="timestamp_ms", dtype=DataType.INT64),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]),
    ]
    collection = Collection(
        collection_name,
        CollectionSchema(fields, description=f"Benchmark-only {index_type} collection"),
    )
    for start in range(0, len(rows), 512):
        batch_rows = rows[start : start + 512]
        batch_vectors = vectors[start : start + 512]
        collection.upsert(
            [
                [row["frame_id"] for row in batch_rows],
                [row["video_id"] for row in batch_rows],
                [int(row["frame_number"]) for row in batch_rows],
                [int(row["timestamp_ms"]) for row in batch_rows],
                [row.get("image_url", "") for row in batch_rows],
                batch_vectors.tolist(),
            ]
        )
    collection.flush()
    collection.create_index(
        "vector",
        {"metric_type": "COSINE", "index_type": index_type, "params": index_params},
    )
    collection.load()
    return collection


def create_hnsw_collection(rows: list[dict[str, Any]], vectors: np.ndarray, collection_name: str, m: int) -> Collection:
    return create_benchmark_collection(
        collection_name,
        rows,
        vectors,
        "HNSW",
        {"M": m, "efConstruction": 256},
    )


def create_ivf_collection(rows: list[dict[str, Any]], vectors: np.ndarray, nlist: int) -> Collection:
    return create_benchmark_collection(BENCH_IVF_COLLECTION, rows, vectors, "IVF_FLAT", {"nlist": nlist})


def create_scann_collection(rows: list[dict[str, Any]], vectors: np.ndarray, nlist: int) -> Collection:
    return create_benchmark_collection(
        BENCH_SCANN_COLLECTION,
        rows,
        vectors,
        "SCANN",
        {"nlist": nlist, "with_raw_data": True},
    )


def ranks_for(results: list[list[str]], query_rows: list[dict[str, Any]]) -> list[int | None]:
    ranks = []
    for result, row in zip(results, query_rows):
        rank = None
        gt_frame_ids = row["gt_frame_ids"]
        for index, frame_id in enumerate(result, start=1):
            if frame_id in gt_frame_ids:
                rank = index
                break
        ranks.append(rank)
    return ranks


def metric_summary(query_rows: list[dict[str, Any]], ranks: list[int | None]) -> dict[str, Any]:
    count = len(query_rows)
    summary: dict[str, Any] = {"count": count}
    for k in [1, 5, 10, 20, 50, 100]:
        summary[f"R@{k}"] = round(sum(rank is not None and rank <= k for rank in ranks) / count * 100, 3)
    summary["MRR@100"] = round(sum((1 / rank) for rank in ranks if rank is not None) / count, 5)
    summary["hits@100"] = sum(rank is not None for rank in ranks)
    hit_ranks = [rank for rank in ranks if rank is not None]
    summary["mean_hit_rank"] = round(mean(hit_ranks), 3) if hit_ranks else None
    summary["misses@100"] = count - summary["hits@100"]
    return summary


def exact_reference_summary(results: list[list[str]], exact_results: list[list[str]]) -> dict[str, Any]:
    if not exact_results:
        return {}
    summary: dict[str, Any] = {}
    for k in [1, 5, 10, 20, 50, 100]:
        scores = []
        for result, exact in zip(results, exact_results):
            result_top = set(result[:k])
            exact_top = set(exact[:k])
            scores.append(len(result_top & exact_top) / max(1, min(k, len(exact_top))))
        summary[f"top{k}_overlap_with_exact_percent"] = round(mean(scores) * 100, 3)
    summary["same_top1_as_exact_percent"] = round(
        sum(bool(result) and bool(exact) and result[0] == exact[0] for result, exact in zip(results, exact_results))
        / len(exact_results)
        * 100,
        3,
    )
    return summary


def summarize_algorithm(
    name: str,
    latencies: list[float],
    results: list[list[str]],
    query_rows: list[dict[str, Any]],
    memory: dict[str, Any],
    extra: dict[str, Any] | None = None,
    exact_results: list[list[str]] | None = None,
) -> dict[str, Any]:
    ranks = ranks_for(results, query_rows)
    summary = {
        "name": name,
        "latency_ms": {
            "p50": round(median(latencies), 3),
            "p95": round(percentile(latencies, 95), 3),
            "p99": round(percentile(latencies, 99), 3),
            "max": round(max(latencies), 3),
        },
        "memory": memory,
        "overall": metric_summary(query_rows, ranks),
        "by_language": {},
        "by_query_type": {},
    }
    if exact_results is not None:
        summary["exact_reference"] = exact_reference_summary(results, exact_results)

    for language in sorted({row["language"] for row in query_rows}):
        indexes = [index for index, row in enumerate(query_rows) if row["language"] == language]
        summary["by_language"][language] = metric_summary(
            [query_rows[index] for index in indexes],
            [ranks[index] for index in indexes],
        )
    for query_type in sorted({row["query_type"] for row in query_rows}):
        indexes = [index for index, row in enumerate(query_rows) if row["query_type"] == query_type]
        summary["by_query_type"][query_type] = metric_summary(
            [query_rows[index] for index in indexes],
            [ranks[index] for index in indexes],
        )
    if extra:
        summary.update(extra)
    return summary


def run_with_memory(name: str, run_fn):
    before_setup = memory_snapshot(f"{name}:before_setup")
    started = time.perf_counter()
    result = run_fn()
    after_search = memory_snapshot(f"{name}:after_search")
    wall_ms = (time.perf_counter() - started) * 1000
    return result, {"before_setup": before_setup, "after_search": after_search, "wall_ms": round(wall_ms, 3)}


def main() -> None:
    args = parse_args()
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")
    milvus_client.connect()

    query_rows = load_queries(args.queries_json)
    query_vectors, encoding_report = encode_queries(query_rows)

    source_collection = milvus_client.collection_name_for_algorithm("flat")
    rows, vectors = load_vectors_from_milvus(source_collection)
    frame_ids = [row["frame_id"] for row in rows]
    nlist = args.ivf_nlist or min(128, max(16, int(math.sqrt(len(rows)) * 2)))
    nprobe = args.ivf_nprobe or min(nlist, 32)
    scann_reorder_k = args.scann_reorder_k or args.top_k * 5

    report = {
        "scope": "PECore text query vectors + vector search; app/reranking/frontend skipped",
        "queries_json": str(args.queries_json),
        "query_count": len(query_rows),
        "dataset": {"vectors": int(vectors.shape[0]), "dim": int(vectors.shape[1]), "source_collection": source_collection},
        "top_k": args.top_k,
        "encoding": encoding_report,
        "memory_tracking": {
            "process_rss_mb": "Python benchmark process RSS from /proc/self/status",
            "gpu": "nvidia-smi memory/utilization snapshot, when available",
            "milvus_container": "docker stats memory snapshot for remote-server-milvus-1, when available",
        },
        "results": [],
        "unavailable": [],
        "benchmark_only_temp_collections": [
            BENCH_HNSW_M16_COLLECTION,
            BENCH_HNSW_M32_COLLECTION,
            BENCH_IVF_COLLECTION,
            BENCH_SCANN_COLLECTION,
        ],
    }

    algorithms = set(args.algorithms)
    exact_results: list[list[str]] | None = None

    if "flat" in algorithms:
        collection = Collection(milvus_client.collection_name_for_algorithm("flat"))
        result, memory = run_with_memory("milvus_flat_exact_cpu", lambda: milvus_search(collection, query_vectors, "flat", args.top_k))
        latencies, results = result
        exact_results = results
        report["results"].append(
            summarize_algorithm(
                "milvus_flat_exact_cpu",
                latencies,
                results,
                query_rows,
                memory,
                {"config": {"index_type": "FLAT"}, "exact_reference": "self"},
            )
        )
    else:
        collection = Collection(milvus_client.collection_name_for_algorithm("flat"))
        _, exact_results = milvus_search(collection, query_vectors, "flat", args.top_k)
        report["exact_reference"] = {
            "name": "milvus_flat_exact_cpu",
            "included_as_result": False,
            "reason": "Exact FLAT is required to compare vector-search algorithms.",
        }

    if "hnsw" in algorithms:
        def run_hnsw_m16():
            started = time.perf_counter()
            collection = create_hnsw_collection(rows, vectors, BENCH_HNSW_M16_COLLECTION, 16)
            build_ms = (time.perf_counter() - started) * 1000
            latencies, results = milvus_search(collection, query_vectors, "hnsw", args.top_k)
            return build_ms, latencies, results

        result, memory = run_with_memory("milvus_hnsw_m16_ef512", run_hnsw_m16)
        build_ms, latencies, results = result
        report["results"].append(summarize_algorithm("milvus_hnsw_m16_ef512", latencies, results, query_rows, memory, {"build_ms": round(build_ms, 3), "config": {"M": 16, "efConstruction": 256, "ef": max(512, args.top_k)}}, exact_results))
        if not args.keep_temp_collections and utility.has_collection(BENCH_HNSW_M16_COLLECTION):
            utility.drop_collection(BENCH_HNSW_M16_COLLECTION)

    if "hnsw_m32" in algorithms:
        def run_hnsw_m32():
            started = time.perf_counter()
            collection = create_hnsw_collection(rows, vectors, BENCH_HNSW_M32_COLLECTION, 32)
            build_ms = (time.perf_counter() - started) * 1000
            latencies, results = milvus_search(collection, query_vectors, "hnsw", args.top_k)
            return build_ms, latencies, results

        result, memory = run_with_memory("milvus_hnsw_m32_ef512", run_hnsw_m32)
        build_ms, latencies, results = result
        report["results"].append(summarize_algorithm("milvus_hnsw_m32_ef512", latencies, results, query_rows, memory, {"build_ms": round(build_ms, 3), "config": {"M": 32, "efConstruction": 256, "ef": max(512, args.top_k)}}, exact_results))
        if not args.keep_temp_collections and utility.has_collection(BENCH_HNSW_M32_COLLECTION):
            utility.drop_collection(BENCH_HNSW_M32_COLLECTION)

    if "ivf_flat" in algorithms:
        try:
            def run_ivf():
                started = time.perf_counter()
                collection = create_ivf_collection(rows, vectors, nlist)
                build_ms = (time.perf_counter() - started) * 1000
                latencies, results = milvus_search(collection, query_vectors, "ivf_flat", args.top_k, nprobe=nprobe)
                return build_ms, latencies, results

            result, memory = run_with_memory("milvus_ivf_flat_cpu", run_ivf)
            build_ms, latencies, results = result
            report["results"].append(summarize_algorithm("milvus_ivf_flat_cpu", latencies, results, query_rows, memory, {"build_ms": round(build_ms, 3), "config": {"nlist": nlist, "nprobe": nprobe}}, exact_results))
        except Exception as exc:
            report["results"].append({"name": "milvus_ivf_flat_cpu", "status": "failed", "error": repr(exc)})
        finally:
            if not args.keep_temp_collections and utility.has_collection(BENCH_IVF_COLLECTION):
                utility.drop_collection(BENCH_IVF_COLLECTION)

    if "gpu_bruteforce" in algorithms:
        try:
            result, memory = run_with_memory("cuvs_gpu_bruteforce_exact", lambda: gpu_bruteforce_search(vectors, frame_ids, query_vectors, args.top_k))
            build_ms, latencies, results = result
            report["results"].append(summarize_algorithm("cuvs_gpu_bruteforce_exact", latencies, results, query_rows, memory, {"build_ms": round(build_ms, 3), "config": {"metric": "inner_product"}}, exact_results))
        except Exception as exc:
            report["results"].append({"name": "cuvs_gpu_bruteforce_exact", "status": "failed", "error": repr(exc)})

    if "cagra" in algorithms:
        try:
            result, memory = run_with_memory("cuvs_cagra_width32", lambda: cagra_search(query_vectors, args.top_k))
            latencies, results = result
            report["results"].append(summarize_algorithm("cuvs_cagra_width32", latencies, results, query_rows, memory, {"config": {"search_width": int(os.getenv("CAGRA_SEARCH_WIDTH", "32")), "graph_degree": 32}}, exact_results))
        except Exception as exc:
            report["results"].append({"name": "cuvs_cagra_width32", "status": "failed", "error": repr(exc)})

    if "gpu_ivf_flat" in algorithms:
        try:
            result, memory = run_with_memory("cuvs_gpu_ivf_flat", lambda: gpu_ivf_search(vectors, frame_ids, query_vectors, args.top_k, nlist, nprobe))
            build_ms, latencies, results = result
            report["results"].append(summarize_algorithm("cuvs_gpu_ivf_flat", latencies, results, query_rows, memory, {"build_ms": round(build_ms, 3), "config": {"n_lists": nlist, "n_probes": nprobe}}, exact_results))
        except Exception as exc:
            report["results"].append({"name": "cuvs_gpu_ivf_flat", "status": "failed", "error": repr(exc)})

    if "scann_raw" in algorithms:
        try:
            def run_scann():
                started = time.perf_counter()
                collection = create_scann_collection(rows, vectors, nlist)
                build_ms = (time.perf_counter() - started) * 1000
                latencies, results = milvus_search(
                    collection,
                    query_vectors,
                    "scann",
                    args.top_k,
                    nprobe=nprobe,
                    reorder_k=scann_reorder_k,
                )
                return build_ms, latencies, results

            result, memory = run_with_memory("milvus_scann_raw_true", run_scann)
            build_ms, latencies, results = result
            report["results"].append(
                summarize_algorithm(
                    "milvus_scann_raw_true",
                    latencies,
                    results,
                    query_rows,
                    memory,
                    {
                        "build_ms": round(build_ms, 3),
                        "config": {
                            "nlist": nlist,
                            "nprobe": nprobe,
                            "with_raw_data": True,
                            "reorder_k": scann_reorder_k,
                        },
                    },
                    exact_results,
                )
            )
        except Exception as exc:
            report["unavailable"].append(
                {
                    "name": "milvus_scann_raw_true",
                    "reason": "Milvus SCANN is unavailable or unsupported in this runtime.",
                    "error": repr(exc),
                }
            )
        finally:
            if not args.keep_temp_collections and utility.has_collection(BENCH_SCANN_COLLECTION):
                utility.drop_collection(BENCH_SCANN_COLLECTION)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
