import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.data_provider import DataProvider


class DataProviderV2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_rejects_unknown_channel(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "ZIP"
        with self.assertRaisesRegex(ValueError, "Unknown channel"):
            await provider.retrieve("made.up", "query", top_k=10)

    async def test_remote_only_channels_say_so_instead_of_returning_nothing(self):
        """An empty list would read as 'no matches' for a channel the lot
        archives simply do not carry."""
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "ZIP"
        for channel in ("subtitled.semantic", "transcript.semantic"):
            with self.assertRaisesRegex(RuntimeError, "remote-server"):
                await provider.retrieve(channel, "query", top_k=10)

    def test_results_carries_hit_fields_and_strips_private_keys(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "ZIP"

        results = provider.results([{
            "frame_id": "L30_V001_000100",
            "video_id": "L30_V001",
            "frame_number": 100,
            "timestamp_ms": 4000,
            "youtube_id": "abcdefghijk",
            "image_url": "/api/zip-frame/L30_V001/4000",
            "score": 0.75,
            "evidence": [{"frame_id": "L30_V001_000100", "_vector": [0.1]}],
        }])

        self.assertEqual(results[0]["youtube_id"], "abcdefghijk")
        self.assertEqual(results[0]["frame_image_url"], "/api/zip-frame/L30_V001/4000")
        self.assertEqual(results[0]["confidence"], 0.75)
        self.assertNotIn("_vector", results[0]["evidence"][0])
        with self.assertRaisesRegex(ValueError, "frame_id"):
            provider.results([{"chunk_id": "chunk-1", "video_id": "v1"}])

    def test_results_uses_the_fps_the_ingest_used(self):
        """timestamp_ms was computed at ingest as frame_number / fps * 1000.
        A hit without fps must get that same number back, not a 25.0 default,
        or the frontend's frame counter drifts on the 91 non-25 fps videos."""
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "ZIP"

        with patch("app.services.zip_frame_source.ingest_fps", return_value=29.97002997002997):
            results = provider.results([{
                "frame_id": "L25_V004_000100", "video_id": "L25_V004",
                "frame_number": 100, "timestamp_ms": 3336, "score": 0.5,
            }])

        self.assertAlmostEqual(results[0]["fps"], 29.97002997002997)

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
                "subtitled.semantic",
                "hello",
                top_k=7,
                video_genre="News",
                exclude_frame_ids=["f0"],
            )

        self.assertEqual(hits[0]["frame_id"], "f1")
        client.__aenter__.return_value.post.assert_awaited_once_with(
            "/api/retrieve",
            json={
                "channel": "subtitled.semantic",
                "query": "hello",
                "top_k": 7,
                "video_genre": "News",
                "exclude_frame_ids": ["f0"],
            },
        )

    async def test_local_mode_fetches_frame_embeddings_through_batch_contract(self):
        provider = DataProvider.__new__(DataProvider)
        provider.mode = "LOCAL"
        response = Mock()
        response.json.return_value = {"embeddings": {"f1": [1.0, 0.0]}}
        response.raise_for_status = lambda: None
        client = AsyncMock()
        client.__aenter__.return_value.post.return_value = response

        with patch("app.data_provider.httpx.AsyncClient", return_value=client):
            embeddings = await provider.frame_embeddings(["f1"])

        self.assertEqual(embeddings, {"f1": [1.0, 0.0]})
        client.__aenter__.return_value.post.assert_awaited_once_with(
            "/api/frame-embeddings",
            json={"frame_ids": ["f1"]},
        )


if __name__ == "__main__":
    unittest.main()
