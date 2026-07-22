import unittest
from unittest.mock import AsyncMock, Mock

from app.strategies._duy_temporal_core import match_temporal
from app.strategies.duy_temporal_search import DuyTemporalSearch


def frame(frame_id, timestamp_ms, score, video_id="v"):
    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_number": timestamp_ms // 40,
        "timestamp_ms": timestamp_ms,
        "image_url": f"/{frame_id}.jpg",
        "score": score,
    }


class DuyTemporalV2Tests(unittest.IsolatedAsyncioTestCase):
    def test_dp_obeys_each_minimum_offset(self):
        rankings = [
            [frame("early", 0, 0.6), frame("late", 4000, 0.9)],
            [frame("end", 5000, 0.8)],
        ]
        groups = [
            {"query": "first", "temporal_offset_ms": 0},
            {"query": "second", "temporal_offset_ms": 5000},
        ]

        hits = match_temporal(rankings, groups)

        self.assertEqual([step["frame_id"] for step in hits[0]["_steps"]], ["early", "end"])
        self.assertAlmostEqual(hits[0]["confidence"], 0.7)

    def test_interval_variants_bound_first_to_last_span(self):
        rankings = [
            [frame("start", 0, 0.5)],
            [
                frame("end4", 4000, 0.5),
                frame("end7", 7000, 0.5),
                frame("end15", 15000, 0.5),
            ],
        ]
        groups = [
            {"query": "first", "temporal_offset_ms": 0},
            {"query": "second", "temporal_offset_ms": 0},
        ]

        self.assertEqual(
            [hit["frame_id"] for hit in match_temporal(rankings, groups, 0, 5000)],
            ["end4"],
        )
        self.assertEqual(
            [hit["frame_id"] for hit in match_temporal(rankings, groups, 5000, 10000)],
            ["end7"],
        )
        self.assertEqual(
            [hit["frame_id"] for hit in match_temporal(rankings, groups, 10000, 20000)],
            ["end15"],
        )

    async def test_strategy_retrieves_one_ranking_per_query_group(self):
        context = Mock()
        context.top_k = 10
        context.query_groups = [
            {"query": "first", "temporal_offset_ms": 0},
            {"query": "second", "temporal_offset_ms": 1000},
        ]
        context.retrieve = AsyncMock(side_effect=[
            [frame("a", 0, 0.8)],
            [frame("b", 1000, 0.7)],
        ])
        context.results.side_effect = lambda hits: [dict(hit) for hit in hits]

        results = await DuyTemporalSearch(Mock()).run(context)

        self.assertEqual(context.retrieve.await_count, 2)
        self.assertEqual([step["frame_id"] for step in results[0]["steps"]], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
