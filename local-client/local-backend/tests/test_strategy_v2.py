import unittest
from unittest.mock import AsyncMock, Mock

from app.strategies._fusion import rrf
from app.strategies.base_strategy import BaseStrategy, SearchContext


class FakeStrategy(BaseStrategy):
    name = "Fake"
    description = "Contract test"
    author = "test"
    version = "2.0"

    async def run(self, context):
        hits = await context.retrieve("raw.semantic", context.query_groups[0]["query"])
        return context.results(hits)


class StrategyV2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_context_delegates_only_public_data_operations(self):
        provider = Mock()
        provider.retrieve = AsyncMock(return_value=[{"frame_id": "f1"}])
        provider.keyframes = AsyncMock(return_value=[{"frame_id": "f2"}])
        provider.results.return_value = [{"frame_id": "f1", "confidence": 0.8}]
        parser = Mock()
        parser.parse_json = AsyncMock(return_value={"visual": "a person"})
        context = SearchContext(
            [{"query": "hello", "temporal_offset_ms": 0}],
            top_k=20,
            video_genre="News",
            data_provider=provider,
            parser=parser,
        )

        hits = await context.retrieve("raw.semantic", "hello")
        frames = await context.keyframes("L01_V001", 1000, 2000, limit=3)
        parsed = await context.parse_json(
            system_prompt="plan",
            user_input="hello",
            response_model=dict,
        )
        results = context.results(hits)

        provider.retrieve.assert_awaited_once_with(
            "raw.semantic", "hello", top_k=20, video_genre="News"
        )
        provider.keyframes.assert_awaited_once_with("L01_V001", 1000, 2000, limit=3)
        self.assertEqual(frames[0]["frame_id"], "f2")
        self.assertEqual(parsed["visual"], "a person")
        self.assertEqual(results[0]["confidence"], 0.8)

    async def test_context_requires_configured_parser(self):
        context = SearchContext([], 10, "All", Mock())
        with self.assertRaisesRegex(RuntimeError, "parser is not configured"):
            await context.parse_json(system_prompt="x", user_input="y", response_model=dict)

    async def test_base_strategy_builds_one_request_context_and_caps_results(self):
        provider = Mock()
        provider.retrieve = AsyncMock(return_value=[{"frame_id": "f1"}])
        provider.results.return_value = [
            {"frame_id": "f1", "confidence": 0.9},
            {"frame_id": "f2", "confidence": 0.8},
        ]

        results = await FakeStrategy(provider).search(
            [{"query": "cat", "temporal_offset_ms": 0}],
            limit=1,
            video_genre="All",
        )

        self.assertEqual([row["frame_id"] for row in results], ["f1"])

    def test_rrf_fuses_rankings_without_adding_raw_scores(self):
        first = [
            {"frame_id": "a", "score": 0.99},
            {"frame_id": "b", "score": 0.80},
        ]
        second = [
            {"frame_id": "b", "score": 12.0},
            {"frame_id": "c", "score": 11.0},
        ]

        fused = rrf([first, second], key="frame_id", k=60)

        self.assertEqual(fused[0]["frame_id"], "b")
        self.assertEqual(len(fused[0]["evidence"]), 2)
        self.assertGreaterEqual(fused[0]["confidence"], fused[1]["confidence"])


if __name__ == "__main__":
    unittest.main()
