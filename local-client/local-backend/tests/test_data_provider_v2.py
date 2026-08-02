import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.data_provider import DataProvider


class DataProviderV2Tests(unittest.IsolatedAsyncioTestCase):
    def test_sample_layout_maps_both_visual_channels_to_same_frame_ids(self):
        provider = DataProvider.__new__(DataProvider)
        provider._feature_paths_by_channel = {}

        videos, frames = provider._load_sample()

        raw = provider._feature_paths_by_channel["raw.semantic"]
        subtitled = provider._feature_paths_by_channel["subtitled.semantic"]
        self.assertTrue(videos)
        self.assertTrue(frames)
        self.assertEqual(set(raw), set(subtitled))
        self.assertEqual(set(raw), {frame["frame_id"] for frame in frames})

    async def test_retrieve_rejects_unknown_channel(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "MOCK"
        with self.assertRaisesRegex(ValueError, "Unknown channel"):
            await provider.retrieve("made.up", "query", top_k=10)

    async def test_keyframes_returns_only_requested_interval(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "SAMPLE"
        provider._frames = [
            {"frame_id": "a", "video_id": "v", "timestamp_ms": 900},
            {"frame_id": "b", "video_id": "v", "timestamp_ms": 1000},
            {"frame_id": "c", "video_id": "v", "timestamp_ms": 2000},
            {"frame_id": "d", "video_id": "v", "timestamp_ms": 2100},
        ]

        frames = await provider.keyframes("v", 1000, 2000, limit=20)

        self.assertEqual([frame["frame_id"] for frame in frames], ["b", "c"])

    def test_results_hydrates_frame_hit_and_rejects_transcript_chunk(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "SAMPLE"
        provider._frames_by_id = {
            "f1": {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_number": 25,
                "timestamp_ms": 1000,
                "image_url": "/static/frames/v1/25.jpg",
            }
        }
        provider._videos_by_id = {
            "v1": {"youtube_id": "abcdefghijk", "fps": 25.0}
        }

        results = provider.results([{
            "frame_id": "f1",
            "score": 0.75,
            "evidence": [{"frame_id": "f1", "_feature_path": "private.npy"}],
        }])

        self.assertEqual(results[0]["youtube_id"], "abcdefghijk")
        self.assertEqual(results[0]["frame_image_url"], "/static/frames/v1/25.jpg")
        self.assertEqual(results[0]["confidence"], 0.75)
        self.assertNotIn("_feature_path", results[0]["evidence"][0])
        with self.assertRaisesRegex(ValueError, "frame_id"):
            provider.results([{"chunk_id": "chunk-1", "video_id": "v1"}])

    async def test_local_mode_proxies_channel_without_exposing_database(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "LOCAL"
        response = Mock()
        response.json.return_value = {"hits": [{"frame_id": "f1"}]}
        response.raise_for_status = lambda: None
        client = AsyncMock()
        client.__aenter__.return_value.post.return_value = response

        with patch("app.data_provider.httpx.AsyncClient", return_value=client):
            hits = await provider.retrieve(
                "subtitled.semantic", "hello", top_k=7, video_genre="News"
            )

        self.assertEqual(hits[0]["frame_id"], "f1")
        client.__aenter__.return_value.post.assert_awaited_once_with(
            "/api/retrieve",
            json={
                "channel": "subtitled.semantic",
                "query": "hello",
                "top_k": 7,
                "video_genre": "News",
            },
        )


if __name__ == "__main__":
    unittest.main()
