import unittest
from unittest.mock import AsyncMock, patch

import main
from app.services.transcript_search import TranscriptSearchService


class TranscriptSearchAlgorithmTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_selection_uses_vector_search(self):
        service = TranscriptSearchService()
        service.search_chunks = AsyncMock(return_value=[{"chunk_id": 1}])

        results = await service.search_algorithm_chunks(
            "xin chao", top_k=5, topic_filter="Thời sự", algorithm="semantic"
        )

        self.assertEqual(results, [{"chunk_id": 1}])
        service.search_chunks.assert_awaited_once_with("xin chao", 5, "Thời sự")

    async def test_lexical_selection_uses_postgres_full_text(self):
        service = TranscriptSearchService()
        rows = [{
            "chunk_id": 2,
            "video_id": "L01_V001",
            "topic": "Thời sự",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "text": "xin chào",
            "score": 0.4,
        }]

        with patch(
            "app.services.transcript_search.postgres_client.search_transcript_chunks_text",
            new=AsyncMock(return_value=rows),
        ) as lexical_search:
            results = await service.search_algorithm_chunks(
                "xin chao", top_k=6, topic_filter="Thời sự", algorithm="lexical"
            )

        self.assertEqual(results[0]["channel"], "transcript.lexical")
        lexical_search.assert_awaited_once_with(
            "xin chao", 6, video_genre="All", topic_filter="Thời sự"
        )

    async def test_fuzzy_selection_uses_rapidfuzz_rows(self):
        service = TranscriptSearchService()
        rows = [{
            "chunk_id": 3,
            "video_id": "L01_V002",
            "topic": "Đời sống",
            "start_time_ms": 1000,
            "end_time_ms": 2000,
            "raw_text": "xin chào các bạn",
        }]

        with patch(
            "app.services.transcript_search.postgres_client.fetch_all_transcript_chunks",
            new=AsyncMock(return_value=rows),
            create=True,
        ) as fetch_rows:
            results = await service.search_algorithm_chunks(
                "xin chao", top_k=4, topic_filter="Đời sống", algorithm="fuzzy"
            )

        self.assertEqual(results[0]["channel"], "transcript.fuzzy")
        self.assertGreaterEqual(results[0]["score"], 0.7)
        fetch_rows.assert_awaited_once_with("Đời sống")

    async def test_unknown_selection_is_rejected(self):
        service = TranscriptSearchService()

        with self.assertRaisesRegex(ValueError, "transcript search algorithm"):
            await service.search_algorithm_chunks("xin chao", algorithm="from-env")

    async def test_request_selection_wins_over_environment(self):
        service = AsyncMock()
        service.search.return_value = []

        with (
            patch.object(main, "_transcript_search_service", service),
            patch.dict("os.environ", {"TRANSCRIPT_SEARCH_ALGORITHM": "fuzzy"}),
        ):
            response = await main.search_transcript(
                main.TranscriptSearchRequest(
                    query="xin chao",
                    top_k=8,
                    topic_filter="Thời sự",
                    algorithm="lexical",
                )
            )

        self.assertEqual(response["algorithm"], "lexical")
        service.search.assert_awaited_once_with(
            "xin chao",
            top_k=8,
            topic_filter="Thời sự",
            algorithm="lexical",
        )


if __name__ == "__main__":
    unittest.main()
