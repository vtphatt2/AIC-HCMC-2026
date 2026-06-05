"""
Run a small set of backend text-to-visual search queries and print top-k hits.

Run from remote-server:
  python scripts/smoke_search_queries.py --top-k 5
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import httpx

logger = logging.getLogger("smoke_search_queries")

DEFAULT_QUERIES = [
    "a busy street with people",
    "a person standing indoors",
    "cars on a road",
    "a building exterior",
    "a close up of food",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test backend Search by Text.")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--strategy-id", default="nam_visual_search_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", action="append", help="Query text. Can be passed multiple times.")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def search_query(
    client: httpx.Client,
    strategy_id: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    response = client.post(
        "/api/search",
        json={
            "strategy_id": strategy_id,
            "query_groups": [{"semantic_query": query, "text_query": "", "temporal_offset_ms": 0}],
            "top_k": top_k,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("results", []))


def print_results(query: str, results: list[dict[str, Any]]) -> None:
    print(f"\nQUERY: {query}")
    print("rank  video_id   frame_id            timestamp_ms  confidence  frame_image_url")
    print("----  ---------  ------------------  ------------  ----------  ----------------")
    for rank, row in enumerate(results, start=1):
        print(
            f"{rank:<4}  "
            f"{str(row.get('video_id', '')):<9}  "
            f"{str(row.get('frame_id', '')):<18}  "
            f"{int(row.get('timestamp_ms') or 0):<12}  "
            f"{float(row.get('confidence') or 0.0):<10.4f}  "
            f"{row.get('frame_image_url', '')}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    queries = args.query or DEFAULT_QUERIES

    logger.info(
        "Running %s smoke queries against %s strategy=%s top_k=%s",
        len(queries),
        args.backend_url,
        args.strategy_id,
        args.top_k,
    )
    with httpx.Client(base_url=args.backend_url.rstrip("/"), timeout=args.timeout) as client:
        for query in queries:
            try:
                results = search_query(client, args.strategy_id, query, args.top_k)
            except httpx.HTTPError as exc:
                logger.exception("Search failed for query=%r: %s", query, exc)
                continue
            print_results(query, results)


if __name__ == "__main__":
    main()
