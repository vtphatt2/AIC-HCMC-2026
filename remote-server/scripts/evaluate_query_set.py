"""
Evaluate a query set without ground truth by logging top-k and score distribution.

Input examples:
  ["a busy street", "cars on a road"]
  {"queries": [{"id": "q1", "semantic_query": "a busy street"}]}

Run from remote-server:
  python scripts/evaluate_query_set.py queries.json --top-k 10
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

logger = logging.getLogger("evaluate_query_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate backend Search by Text query set.")
    parser.add_argument("queries_json", type=Path)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--strategy-id", default="nam_visual_search_v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=Path("evaluation_report.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("evaluation_report.csv"))
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_queries = payload.get("queries", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_queries, list):
        raise ValueError("queries_json must contain a list or an object with a 'queries' list")

    queries = []
    for index, item in enumerate(raw_queries, start=1):
        if isinstance(item, str):
            queries.append({"id": f"q{index}", "query_groups": [{"semantic_query": item, "text_query": "", "temporal_offset_ms": 0}]})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported query item at index {index}: {item!r}")

        query_id = str(item.get("id") or f"q{index}")
        if "query_groups" in item:
            query_groups = item["query_groups"]
        else:
            query_groups = [
                {
                    "semantic_query": str(item.get("semantic_query") or item.get("query") or ""),
                    "text_query": str(item.get("text_query") or ""),
                    "temporal_offset_ms": int(item.get("temporal_offset_ms") or 0),
                }
            ]
        queries.append({"id": query_id, "query_groups": query_groups})
    return queries


def score_distribution(results: list[dict[str, Any]]) -> dict[str, float]:
    scores = [float(row.get("confidence") or 0.0) for row in results]
    if not scores:
        return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0}
    return {
        "count": float(len(scores)),
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.fmean(scores),
        "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    }


def run_query(
    client: httpx.Client,
    strategy_id: str,
    query_groups: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    response = client.post(
        "/api/search",
        json={"strategy_id": strategy_id, "query_groups": query_groups, "top_k": top_k},
    )
    response.raise_for_status()
    return dict(response.json())


def write_csv(path: Path, report: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "query_id",
                "rank",
                "video_id",
                "frame_id",
                "timestamp_ms",
                "confidence",
                "frame_image_url",
            ],
        )
        writer.writeheader()
        for query_report in report:
            for rank, row in enumerate(query_report["results"], start=1):
                writer.writerow(
                    {
                        "query_id": query_report["id"],
                        "rank": rank,
                        "video_id": row.get("video_id", ""),
                        "frame_id": row.get("frame_id", ""),
                        "timestamp_ms": row.get("timestamp_ms", ""),
                        "confidence": row.get("confidence", ""),
                        "frame_image_url": row.get("frame_image_url", ""),
                    }
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    queries = load_queries(args.queries_json)
    report: list[dict[str, Any]] = []

    logger.info("Evaluating %s queries against %s", len(queries), args.backend_url)
    with httpx.Client(base_url=args.backend_url.rstrip("/"), timeout=args.timeout) as client:
        for query in tqdm(queries, desc="Evaluating queries", unit="query"):
            query_id = str(query["id"])
            try:
                payload = run_query(client, args.strategy_id, query["query_groups"], args.top_k)
                results = list(payload.get("results", []))
                distribution = score_distribution(results)
                logger.info(
                    "%s: results=%s mean=%.4f max=%.4f",
                    query_id,
                    len(results),
                    distribution["mean"],
                    distribution["max"],
                )
                report.append(
                    {
                        "id": query_id,
                        "query_groups": query["query_groups"],
                        "strategy_id": args.strategy_id,
                        "execution_time_ms": payload.get("execution_time_ms"),
                        "score_distribution": distribution,
                        "results": results,
                    }
                )
            except Exception as exc:
                logger.exception("Query failed id=%s: %s", query_id, exc)
                report.append({"id": query_id, "query_groups": query["query_groups"], "error": str(exc), "results": []})

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"queries": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, report)
    logger.info("Wrote %s and %s", args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
