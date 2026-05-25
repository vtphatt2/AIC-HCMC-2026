"""
Mock strategy test suite.

Run after starting the local backend in MOCK mode:
    uvicorn main:app --reload --port 8000

Usage:
    python scripts/test_strategy_mock.py
    python scripts/test_strategy_mock.py --strategy nam_defensive_fusion_v1
"""

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

API = "http://localhost:8000"


def search(strategy_id: str, query_groups: list[dict], top_k: int = 10) -> dict:
    """Call POST /api/search and return the parsed JSON response."""
    body = json.dumps({
        "strategy_id": strategy_id,
        "query_groups": query_groups,
        "top_k": top_k,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API}/api/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}: {detail}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}


def get_strategies() -> list[str]:
    """Return list of available strategy IDs from /api/strategies."""
    try:
        with urllib.request.urlopen(f"{API}/api/strategies", timeout=5) as resp:
            return [s["id"] for s in json.loads(resp.read())]
    except Exception:
        return []


# ---------------------------------------------------------------------------
#  Test case definitions: (name, query_groups, expected_behavior, assertions)
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ── Semantic-only queries ────────────────────────────────────────────
    {
        "name": "Semantic-only: visual query",
        "query_groups": [
            {"semantic_query": "a city street with floodwater", "text_query": "", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "Returns up to 10 results sorted by confidence descending. "
            "First result should come from dQw4w9WgXcQ (city street video). "
            "All results have the 7 required keys. "
            "Confidence values are in [0, 1]. "
            "No score field from MOCK mode — strategy uses fallback 0.5."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
            lambda r: len(r["results"]) <= 10,
            lambda r: all(0.0 <= x["confidence"] <= 1.0 for x in r["results"]),
            lambda r: all(
                k in x for x in r["results"]
                for k in ("video_id", "frame_id", "frame_number", "timestamp_ms", "confidence", "frame_image_url", "fps")
            ),
        ],
    },
    # ── Text-only queries (OCR) ───────────────────────────────────────────
    {
        "name": "Text-only: OCR exact match — GOAL",
        "query_groups": [
            {"semantic_query": "", "text_query": "GOAL", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "With OCR-only strategy, the top result should be 6stlCkUDG_s_000060 "
            "(OCR text contains 'GOAL! Vietnam 1 - 0 Thailand'). "
            "OCR-matched frames should rank above non-OCR frames."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
            lambda r: any(
                "6stlCkUDG_s" in x["video_id"] for x in r["results"][:3]
            ),
        ],
    },
    {
        "name": "Text-only: OCR exact match — BREAKING",
        "query_groups": [
            {"semantic_query": "", "text_query": "BREAKING", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "dQw4w9WgXcQ_000025 should rank highly — OCR text is "
            "'BREAKING NEWS: City Center Flooding'. "
            "Also, dQw4w9WgXcQ_009000 (EMERGENCY BROADCAST) should appear."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
            lambda r: any(
                "dQw4w9WgXcQ" in x["video_id"] for x in r["results"][:5]
            ),
        ],
    },
    {
        "name": "Text-only: OCR exact match — pangolin (unique keyword)",
        "query_groups": [
            {"semantic_query": "", "text_query": "pangolin", "temporal_offset_ms": 0},
        ],
        "top_k": 5,
        "expect": (
            "Only jNQXAC9IVRw_000450 has 'pangolin' in OCR text. "
            "That frame should rank #1 with highest confidence. "
            "No other frame should have OCR match for this keyword."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
            lambda r: r["results"][0]["video_id"] == "jNQXAC9IVRw",
        ],
    },
    # ── Text-only queries (transcripts) ───────────────────────────────────
    {
        "name": "Text-only: transcript match — Mekong Delta",
        "query_groups": [
            {"semantic_query": "", "text_query": "Mekong Delta", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "Transcript for jNQXAC9IVRw contains 'Mekong Delta' at 0–3500ms. "
            "Frames in that time range should rank higher. "
            "Also OCR frame jNQXAC9IVRw_000030 has 'Mekong Delta Wildlife Reserve'."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
        ],
    },
    {
        "name": "Text-only: transcript match — champions",
        "query_groups": [
            {"semantic_query": "", "text_query": "Champions", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "Transcript for 6stlCkUDG_s contains '2026 AFF Champions' at 9000–12000ms. "
            "OCR also has 'AFF Championship 2026' at 9000ms. "
            "Both modalities should boost 6stlCkUDG_s frames near 9000ms."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
        ],
    },
    # ── Mixed queries (semantic + text) ───────────────────────────────────
    {
        "name": "Mixed: visual + OCR — 'flood' + 'emergency'",
        "query_groups": [
            {"semantic_query": "flooded city street", "text_query": "emergency", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "OCR: 'Mayor announces emergency response plan' (dQw4w9WgXcQ_000075) "
            "and 'EMERGENCY BROADCAST' (dQw4w9WgXcQ_009000). "
            "Combined with visual query about flooding — dQw4w9WgXcQ frames should dominate top results."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
        ],
    },
    # ── Empty query ───────────────────────────────────────────────────────
    {
        "name": "Empty query: both fields blank",
        "query_groups": [
            {"semantic_query": "", "text_query": "", "temporal_offset_ms": 0},
        ],
        "top_k": 5,
        "expect": (
            "Should still return results (fallback to visual with default score 0.5). "
            "All results should have equal or near-equal confidence. "
            "Must not crash or return empty."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
            lambda r: len(r["results"]) <= 5,
        ],
    },
    # ── Temporal multi-step ──────────────────────────────────────────────
    {
        "name": "Temporal: two-step — GOAL then crowd",
        "query_groups": [
            {"semantic_query": "goal celebration", "text_query": "GOAL", "temporal_offset_ms": 0},
            {"semantic_query": "crowd cheering", "text_query": "Champions", "temporal_offset_ms": 4000},
        ],
        "top_k": 10,
        "expect": (
            "Step 1 should find GOAL frames (6stlCkUDG_s_000060 at 1000ms). "
            "Step 2 should find frames ~4000ms later (~5000ms) in same video. "
            "6stlCkUDG_s_000300 (5000ms, 'Half-time score') is near the target. "
            "Temporal constraint should favor 6stlCkUDG_s over other videos."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
        ],
    },
    # ── Duplicate suppression ─────────────────────────────────────────────
    {
        "name": "Dedup: dense cluster — very close frames",
        "query_groups": [
            {"semantic_query": "", "text_query": "emergency", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "dQw4w9WgXcQ has dense frames at 360000, 360200, 360400ms (200ms gaps). "
            "A strategy with dedup (min 500ms gap) should suppress at least 1 of the 3. "
            "Frames dQw4w9WgXcQ_009005 and dQw4w9WgXcQ_009010 may be dropped. "
            "Without dedup, all 3 appear consecutively."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
            lambda r: len(r["results"]) <= 10,
        ],
    },
    # ── Missing field handling ────────────────────────────────────────────
    {
        "name": "Missing field: frame without image_url",
        "query_groups": [
            {"semantic_query": "", "text_query": "broadcast", "temporal_offset_ms": 0},
        ],
        "top_k": 10,
        "expect": (
            "dQw4w9WgXcQ_009999 has no image_url field. "
            "Strategy should handle this gracefully (empty string default). "
            "Must not KeyError. Other 7 required keys should still be present."
        ),
        "asserts": [
            lambda r: len(r["results"]) > 0,
            lambda r: all(
                k in x for x in r["results"]
                for k in ("video_id", "frame_id", "frame_number", "timestamp_ms", "confidence", "frame_image_url", "fps")
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
#  Test runner
# ---------------------------------------------------------------------------

def run_tests(strategy_ids: list[str]):
    if not strategy_ids:
        print("No strategies found. Is the local backend running?\n")
        print("Start it with:  uvicorn main:app --reload --port 8000")
        sys.exit(1)

    passed = 0
    failed = 0

    for sid in strategy_ids:
        print(f"\n{'='*70}")
        print(f"  Strategy: {sid}")
        print(f"{'='*70}")

        for tc in TEST_CASES:
            print(f"\n  [{tc['name']}]")
            print(f"  Expected: {tc['expect']}")

            resp = search(sid, tc["query_groups"], tc.get("top_k", 10))

            if "error" in resp:
                print(f"  [FAIL] ERROR: {resp['error']}")
                failed += 1
                continue

            n = resp.get("total", len(resp.get("results", [])))
            elapsed = resp.get("execution_time_ms", "?")
            print(f"  -> {n} results in {elapsed}ms")

            results = resp.get("results", [])
            if results:
                top = results[0]
                print(f"  Top: {top['video_id']} | {top['frame_id']} "
                      f"| conf={top['confidence']:.4f} | ts={top['timestamp_ms']}ms")
                # Show top 3
                for i, r in enumerate(results[:3], 1):
                    print(f"    #{i}: {r['frame_id']} conf={r['confidence']:.4f}")

            # Run assertions
            tc_passed = True
            for i, check in enumerate(tc["asserts"]):
                try:
                    if not check(resp):
                        print(f"  [FAIL] Assertion #{i+1} FAILED")
                        tc_passed = False
                except Exception as exc:
                    print(f"  [FAIL] Assertion #{i+1} ERROR: {exc}")
                    tc_passed = False

            if tc_passed:
                print(f"  [PASS] All assertions passed")
                passed += 1
            else:
                failed += 1

    print(f"\n{'='*70}")
    print(f"  Summary: {passed} passed, {failed} failed out of {len(TEST_CASES) * len(strategy_ids)}")
    print(f"{'='*70}\n")


def main():
    # Check backend is running
    try:
        urllib.request.urlopen(f"{API}/api/health", timeout=3)
    except Exception:
        print(f"ERROR: Cannot reach {API}/api/health")
        print("Start the local backend first:")
        print("  cd local-client/local-backend")
        print("  uvicorn main:app --reload --port 8000")
        sys.exit(1)

    # Determine which strategies to test
    args = sys.argv[1:]
    if "--strategy" in args:
        idx = args.index("--strategy")
        strategy_ids = [args[idx + 1]] if idx + 1 < len(args) else []
    else:
        strategy_ids = get_strategies()

    if not strategy_ids:
        print("No strategies available. Available:")
        print(json.dumps(get_strategies(), indent=2))
        sys.exit(1)

    print(f"Testing strategies: {strategy_ids}")
    print(f"Mock data: {Path(__file__).parent.parent / 'app' / 'mock'}")
    run_tests(strategy_ids)


if __name__ == "__main__":
    main()
