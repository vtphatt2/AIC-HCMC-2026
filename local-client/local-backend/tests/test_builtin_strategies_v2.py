import unittest
from unittest.mock import AsyncMock, Mock

from app.strategies.raw_visual import RawVisual
from app.strategies.multi_source import MultiSource
from app.strategies.multi_source_temporal import MultiSourceTemporal
from app.strategies.temporal_visual import TemporalVisual


class BuiltinStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_visual_uses_raw_query_and_channel(self):
        context = Mock()
        context.query_groups = [{"query": "red bus", "temporal_offset_ms": 0}]
        context.retrieve = AsyncMock(return_value=[{"frame_id": "f1"}])
        context.results.return_value = [{"frame_id": "f1"}]

        results = await RawVisual(Mock()).run(context)

        context.retrieve.assert_awaited_once_with("raw.semantic", "red bus")
        self.assertEqual(results[0]["frame_id"], "f1")

    async def test_temporal_visual_keeps_rankings_separate(self):
        context = Mock()
        context.top_k = 10
        context.query_groups = [
            {"query": "door opens", "temporal_offset_ms": 0},
            {"query": "person exits", "temporal_offset_ms": 5000},
        ]
        context.retrieve = AsyncMock(side_effect=[
            [{"frame_id": "a", "video_id": "v", "timestamp_ms": 1000, "score": 0.9}],
            [{"frame_id": "b", "video_id": "v", "timestamp_ms": 6000, "score": 0.8}],
        ])
        context.option.return_value = [1, 3]
        context.results.side_effect = lambda hits: [dict(hit) for hit in hits]

        results = await TemporalVisual(Mock()).run(context)

        self.assertEqual(context.retrieve.await_count, 2)
        self.assertEqual(results[0]["frame_id"], "b")
        self.assertAlmostEqual(results[0]["confidence"], 0.825)
        self.assertEqual([step["frame_id"] for step in results[0]["steps"]], ["a", "b"])

    async def test_multi_source_explicitly_maps_transcript_intervals_to_frames(self):
        context = Mock()
        context.top_k = 10
        context.option.side_effect = lambda _key, default: default
        context.query_groups = [{"query": "market", "temporal_offset_ms": 0}]
        context.retrieve = AsyncMock(side_effect=[
            [],
            [],
            [{
                "channel": "transcript.lexical",
                "chunk_id": "c1",
                "video_id": "v",
                "start_time_ms": 1000,
                "end_time_ms": 2000,
                "score": 0.8,
            }],
            RuntimeError("semantic index absent"),
        ])
        context.keyframes = AsyncMock(return_value=[
            {"frame_id": "f", "video_id": "v", "timestamp_ms": 1500}
        ])
        context.results.side_effect = lambda hits: hits

        results = await MultiSource(Mock()).run(context)

        context.keyframes.assert_awaited_once_with("v", 1000, 2000, limit=3)
        self.assertEqual(results[0]["frame_id"], "f")
        self.assertEqual(results[0]["evidence"][0]["chunk_id"], "c1")

    async def test_multi_source_temporal_fuses_each_event_before_matching(self):
        context = Mock()
        context.top_k = 10
        context.query_groups = [
            {"query": "door opens", "temporal_offset_ms": 0},
            {"query": "person exits", "temporal_offset_ms": 1000},
        ]

        async def retrieve(channel, query, *, top_k):
            if channel != "raw.semantic":
                return []
            timestamp = 0 if query == "door opens" else 1000
            return [{
                "frame_id": query,
                "video_id": "v",
                "timestamp_ms": timestamp,
                "score": 0.8,
            }]

        context.retrieve = AsyncMock(side_effect=retrieve)
        context.keyframes = AsyncMock(return_value=[])
        context.option.side_effect = lambda key, default: [1, 2] if key == "event_weights" else default
        context.results.side_effect = lambda hits: [dict(hit) for hit in hits]

        results = await MultiSourceTemporal(Mock()).run(context)

        self.assertEqual(context.retrieve.await_count, 8)
        self.assertEqual([step["frame_id"] for step in results[0]["steps"]], ["door opens", "person exits"])


if __name__ == "__main__":
    unittest.main()
