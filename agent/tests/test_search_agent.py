import json
import unittest

import httpx

from agent.models import SearchAttempt, SearchEvent, SearchPlan
from agent.search_agent import SearchAgent
from agent.tools import VortaClient


class FakePlanner:
    def __init__(self, plan: SearchPlan):
        self.plan_value = plan
        self.calls = []

    def plan(self, query, capabilities, cancel_event=None):
        self.calls.append((query, capabilities))
        return self.plan_value, "thread-test"


def _plan(*, fallback=False):
    fallback_attempt = None
    if fallback:
        fallback_attempt = SearchAttempt(
            mode="frames",
            strategy="duy_multi_detail_search",
            events=[SearchEvent(visual="red shirt man sorting mangoes")],
        )
    return SearchPlan(
        plan_summary="Search the visually distinctive action first.",
        primary=SearchAttempt(
            mode="frames",
            strategy="raw_visual",
            events=[SearchEvent(visual="man in a red shirt sorting mangoes")],
        ),
        fallback=fallback_attempt,
    )


def _transport(search_responses):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "env_mode": "ZIP"})
        if request.url.path == "/api/strategies":
            return httpx.Response(200, json=[
                {"id": "raw_visual", "name": "Raw visual", "description": "visual"},
                {"id": "duy_multi_detail_search", "name": "Multi detail", "description": "details"},
            ])
        if request.url.path == "/api/transcript-search-algorithms":
            return httpx.Response(200, json={"default": "fuzzy", "algorithms": []})
        if request.url.path == "/api/search":
            payload = search_responses.pop(0)
            return httpx.Response(200, json=payload)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return httpx.MockTransport(handler), calls


class SearchAgentTests(unittest.TestCase):
    def test_uses_structured_plan_and_existing_search_contract(self):
        transport, calls = _transport([{
            "results": [{
                "video_id": "L21_V011",
                "frame_id": "L21_V011_022824",
                "frame_number": 22824,
                "timestamp_ms": 912960,
                "confidence": 0.22,
                "frame_image_url": "/api/zip-frame/L21_V011/912960",
                "fps": 25.0,
            }],
            "total": 1,
            "strategy_id": "raw_visual",
            "execution_time_ms": 10,
        }])
        planner = FakePlanner(_plan())

        with httpx.Client(transport=transport, base_url="http://vorta.test") as http:
            result = SearchAgent(planner, VortaClient("http://vorta.test", http=http)).run(
                "a man in a red shirt sorts mangoes", top_k=5
            )

        self.assertEqual(result["thread_id"], "thread-test")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["results"][0]["frame_id"], "L21_V011_022824")
        search_request = next(call for call in calls if call.url.path == "/api/search")
        payload = json.loads(search_request.content)
        self.assertEqual(payload, {
            "strategy_id": "raw_visual",
            "query_groups": [{
                "query": "man in a red shirt sorting mangoes",
                "temporal_offset_ms": 0,
            }],
            "top_k": 5,
        })

    def test_uses_at_most_one_fallback_when_first_search_is_empty(self):
        transport, calls = _transport([
            {"results": [], "total": 0, "strategy_id": "raw_visual", "execution_time_ms": 7},
            {
                "results": [{"video_id": "L22_V002", "frame_id": "L22_V002_000100"}],
                "total": 1,
                "strategy_id": "duy_multi_detail_search",
                "execution_time_ms": 8,
            },
        ])

        with httpx.Client(transport=transport, base_url="http://vorta.test") as http:
            result = SearchAgent(
                FakePlanner(_plan(fallback=True)),
                VortaClient("http://vorta.test", http=http),
            ).run("red shirt man sorting mangoes", top_k=10)

        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["results"][0]["video_id"], "L22_V002")
        self.assertEqual(sum(call.url.path == "/api/search" for call in calls), 2)

    def test_does_not_run_fallback_after_useful_results(self):
        transport, calls = _transport([{
            "results": [{"video_id": "L21_V011", "frame_id": "L21_V011_022824"}],
            "total": 1,
            "strategy_id": "raw_visual",
            "execution_time_ms": 5,
        }])

        with httpx.Client(transport=transport, base_url="http://vorta.test") as http:
            result = SearchAgent(
                FakePlanner(_plan(fallback=True)),
                VortaClient("http://vorta.test", http=http),
            ).run("red shirt man sorting mangoes", top_k=10)

        self.assertEqual(result["attempts"], 1)
        self.assertEqual(sum(call.url.path == "/api/search" for call in calls), 1)

    def test_plan_rejects_more_than_four_events(self):
        with self.assertRaises(ValueError):
            SearchPlan(
                plan_summary="too broad",
                primary=SearchAttempt(
                    mode="frames",
                    strategy="temporal_visual",
                    events=[SearchEvent(visual=f"event {index}") for index in range(5)],
                ),
                fallback=None,
            )


if __name__ == "__main__":
    unittest.main()
