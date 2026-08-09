import math
import unittest
from unittest.mock import AsyncMock, Mock

from app.strategies._similarity_filter import filter_similar_results
from app.strategies.base_strategy import BaseStrategy


def vector(cosine):
    return [cosine, math.sqrt(1 - cosine * cosine)]


def result(frame_id, *, steps=None):
    row = {"frame_id": frame_id, "confidence": 1.0}
    if steps is not None:
        row["steps"] = [{"frame_id": item} for item in steps]
    return row


class SimilarityFilterTests(unittest.TestCase):
    def test_temporal_result_is_removed_only_when_every_step_is_similar(self):
        candidates = [
            result("a2", steps=["a1", "a2"]),
            result("b2", steps=["b1", "b2"]),
            result("c2", steps=["c1", "c2"]),
        ]
        embeddings = {
            "a1": [1.0, 0.0], "a2": [1.0, 0.0],
            "b1": vector(0.99), "b2": vector(0.99),
            "c1": vector(0.99), "c2": vector(0.90),
        }

        kept = filter_similar_results(candidates, embeddings, threshold=0.98)

        self.assertEqual([item["frame_id"] for item in kept], ["a2", "c2"])

    def test_event_weight_scales_each_steps_cosine_distance(self):
        candidates = [result("a", steps=["a"]), result("b", steps=["b"])]
        embeddings = {"a": [1.0, 0.0], "b": vector(0.99)}

        kept = filter_similar_results(
            candidates,
            embeddings,
            threshold=0.985,
            event_weights=[2.0],
        )

        self.assertEqual([item["frame_id"] for item in kept], ["a", "b"])

    def test_similarity_equal_to_threshold_is_not_removed(self):
        candidates = [result("a"), result("b")]
        embeddings = {"a": [1.0, 0.0], "b": vector(0.98)}

        kept = filter_similar_results(candidates, embeddings, threshold=0.98)

        self.assertEqual([item["frame_id"] for item in kept], ["a", "b"])


class FilteredStrategy(BaseStrategy):
    name = "Filtered test"
    description = "Test strategy"
    author = "Tests"

    async def run(self, context):
        return context.results(await context.retrieve("raw.semantic", "query", top_k=2))


class SimilarityBackfillTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_backfills_with_database_exclusion_after_filtering(self):
        first_page = [
            {"frame_id": "a", "score": 1.0},
            {"frame_id": "b", "score": 0.9},
        ]
        second_page = [{"frame_id": "c", "score": 0.8}]
        provider = Mock()
        provider.retrieve = AsyncMock(side_effect=[first_page, second_page])
        provider.results.side_effect = lambda hits: [dict(hit) for hit in hits]
        provider.frame_embeddings = AsyncMock(return_value={
            "a": [1.0, 0.0],
            "b": vector(0.99),
            "c": [0.0, 1.0],
        })

        results = await FilteredStrategy(provider).search(
            [{"query": "query"}],
            limit=2,
            duplicate_threshold=0.98,
        )

        self.assertEqual([item["frame_id"] for item in results], ["a", "c"])
        self.assertEqual(provider.retrieve.await_count, 2)
        self.assertEqual(provider.frame_embeddings.await_args_list[1].args[0], ["c"])
        self.assertEqual(
            provider.retrieve.await_args_list[1].kwargs["exclude_frame_ids"],
            ["a", "b"],
        )


if __name__ == "__main__":
    unittest.main()
