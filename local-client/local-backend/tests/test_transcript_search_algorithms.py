import unittest
from unittest.mock import AsyncMock, patch

from app.data_provider import DataProvider
from app.services import transcript_index


class TranscriptSearchAlgorithmTests(unittest.IsolatedAsyncioTestCase):
    async def test_zip_mode_routes_explicit_fuzzy_selection(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "ZIP"

        with patch(
            "app.services.transcript_index.search_all_transcripts",
            return_value=[{"chunk_id": 1}],
        ) as fuzzy_search:
            results = await provider.search_transcript_chunks(
                "xin chao", limit=7, algorithm="fuzzy"
            )

        self.assertEqual(results, [{"chunk_id": 1}])
        fuzzy_search.assert_called_once_with("xin chao", top_k=7)

    async def test_zip_mode_routes_explicit_lexical_selection(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "ZIP"

        with patch(
            "app.services.transcript_index.search_all_transcripts_lexical",
            return_value=[{"chunk_id": 2}],
            create=True,
        ) as lexical_search:
            results = await provider.search_transcript_chunks(
                "xin chao", limit=9, algorithm="lexical"
            )

        self.assertEqual(results, [{"chunk_id": 2}])
        lexical_search.assert_called_once_with("xin chao", top_k=9)

    async def test_local_mode_forwards_user_selection_to_remote(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "LOCAL"
        provider._transcript_chunks_search_remote = AsyncMock(return_value=[])

        await provider.search_transcript_chunks(
            "xin chao", limit=11, topic_filter="Thời sự", algorithm="lexical"
        )

        provider._transcript_chunks_search_remote.assert_awaited_once_with(
            "xin chao", 11, "Thời sự", "lexical"
        )

    async def test_unknown_algorithm_is_rejected_instead_of_falling_back(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "ZIP"

        with self.assertRaisesRegex(ValueError, "transcript search algorithm"):
            await provider.search_transcript_chunks("xin chao", algorithm="from-env")

    def test_local_lexical_search_ranks_by_exact_query_token_coverage(self):
        segments = [
            ("L01_V001", transcript_index.TranscriptSegment(0, 1000, "xin chào bạn")),
            ("L01_V002", transcript_index.TranscriptSegment(0, 1000, "xin mọi người")),
        ]

        with (
            patch.object(transcript_index, "_all_segments", return_value=segments),
            patch.object(
                transcript_index,
                "_transcript_matches_to_results",
                side_effect=lambda matches: matches,
            ),
        ):
            matches = transcript_index.search_all_transcripts_lexical("xin chào", top_k=2)

        self.assertEqual([video_id for video_id, _match in matches], ["L01_V001", "L01_V002"])
        self.assertEqual(matches[0][1].score, 100.0)
        self.assertEqual(matches[1][1].score, 50.0)


if __name__ == "__main__":
    unittest.main()
