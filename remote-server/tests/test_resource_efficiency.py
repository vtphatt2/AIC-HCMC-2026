"""Keep frame identity and transcript behavior while reducing resident data."""
from array import array
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import main
from app.data_provider import DataProvider
from app.services.local_zip_media import VideoFrameIndex, _range_plan, REORDER_LOOKAHEAD
from app.services.transcript_search import TranscriptSearchService


def scalar_plan(index, frame_id, margin=4):
    """Original scalar rule, including presentation-order ties and B-frames."""
    target = index.base_pts + frame_id / index.fps * index.timescale if index.fps else 0
    start = index.nearest_keyframe_sample(frame_id)
    pts = index.sample_pts
    selected = min(range(start, len(pts)), key=lambda i: (abs(pts[i] - target), i))
    horizon = min(len(pts), selected + 1 + REORDER_LOOKAHEAD)
    last = max(i for i in range(start, horizon) if pts[i] <= pts[selected])
    end = min(last + margin, len(pts) - 1)
    return start, selected, end, sum(pts[i] < pts[selected] for i in range(start, end + 1))


class CompactFrameIndexTests(unittest.TestCase):
    def index(self, pts, fps=25., timescale=1000, keys=None):
        keys = keys or [0]
        base = min(pts)
        return VideoFrameIndex(
            zip_path=Path("unused.zip"), data_offset=2**33, timescale=timescale,
            fps=fps, base_pts=base, sample_offsets=array("Q", (2**34+i*100 for i in range(len(pts)))),
            sample_sizes=array("Q", [100]*len(pts)), sample_pts=array("q", pts),
            keyframe_samples=keys,
            keyframe_frames=[round((pts[i]-base)/timescale*fps) if fps else i for i in keys],
        )

    def test_reordered_frames_and_fractional_fps_match_scalar_selection(self):
        rng = random.Random(2026)
        pts = list(range(0, 400*1001, 1001))
        for start in range(0, len(pts), 8):
            block = pts[start:start+8]
            rng.shuffle(block)
            pts[start:start+8] = block
        for fps in [25., 30000/1001, 30., 60.]:
            index = self.index(pts, fps, 30000, [0, 128, 256])
            for frame in [-1, 0, 1, 399, 10000] + [rng.randrange(800) for _ in range(200)]:
                self.assertEqual(_range_plan(index, frame), scalar_plan(index, frame))

    def test_ties_duplicate_pts_negative_pts_and_large_values(self):
        for pts in [[-80, 0, -40, 40, 40, 120, 80], [0, 80, 40, 160, 120],
                    [2**54+4, 2**54, 2**54+2, 2**54+8]]:
            for fps in [0., 25., 30000/1001]:
                index = self.index(pts, fps)
                for frame in range(-2, 30):
                    self.assertEqual(_range_plan(index, frame), scalar_plan(index, frame))

    def test_columns_retain_wide_offsets_and_signed_timestamps(self):
        index = self.index([-2**40, 0, 2**40])
        self.assertEqual(list(index.sample_offsets), [2**34, 2**34+100, 2**34+200])
        self.assertEqual(list(index.sample_pts), [-2**40, 0, 2**40])
        self.assertEqual(np.frombuffer(index.sample_pts, dtype=np.int64).tolist(), list(index.sample_pts))


class FrameEmbeddingJsonTests(unittest.IsolatedAsyncioTestCase):
    async def test_numeric_arrays_remain_exact_json_lists(self):
        import httpx
        values = np.array([0., -0., 1e-38, -.999999, 1e30], dtype=np.float32)
        provider = MagicMock()
        provider.frame_embeddings = AsyncMock(return_value={"a": values, "b": list(values)})
        with patch.object(main, "_data_provider", provider):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                response = await client.post("/api/frame-embeddings", json={"frame_ids": ["a", "b", "missing"]})
        self.assertEqual(response.status_code, 200)
        actual = response.json()["embeddings"]
        self.assertEqual(list(actual), ["a", "b"])
        for vector in actual.values():
            self.assertIsInstance(vector, list)
            self.assertEqual(np.asarray(vector, dtype=np.float32).tobytes(), values.tobytes())

    async def test_empty_and_oversized_requests_keep_contract(self):
        import httpx
        provider = MagicMock()
        provider.frame_embeddings = AsyncMock(return_value={})
        with patch.object(main, "_data_provider", provider):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                response = await client.post("/api/frame-embeddings", json={"frame_ids": []})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"embeddings": {}})
                response = await client.post("/api/frame-embeddings", json={"frame_ids": ["a"] * 20001})
                self.assertEqual(response.status_code, 400)
        provider.frame_embeddings.assert_awaited_once_with([])


class SharedTranscriptServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_fallback_selects_real_jpeg_served_by_static_mount(self):
        import httpx
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory) / "test_video"
            frame_dir.mkdir()
            for frame, color in [(100, "red"), (200, "blue")]:
                Image.new("RGB", (8, 8), color).save(frame_dir / f"{frame:06d}.jpg")
            with (
                patch.dict("os.environ", {"FRAME_STATIC_DIR": f" {directory} "}),
                patch("app.data_provider.TRANSCRIPT_CHUNK_SEARCH_ENABLED", True),
                patch("app.data_provider.PECoreTextEncoder"),
                patch.object(DataProvider, "_get_collection"),
            ):
                provider = DataProvider()
                self.addCleanup(provider.close)
                service = provider.transcript_search_service
                self.addCleanup(service._nearest_frame.cache_clear)
                frame = service._nearest_frame("test_video", 4200)
                self.assertEqual(frame["frame_number"], 100)
                self.assertEqual(frame["timestamp_ms"], 4000)
                app = FastAPI()
                app.mount("/static/frames", StaticFiles(directory=main._frame_static_dir()))
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.get(frame["image_url"])
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "image/jpeg")
                self.assertEqual(response.content, (frame_dir / "000100.jpg").read_bytes())

    async def test_shared_service_expands_configured_frame_directory_like_api(self):
        for configured in ["~", " ~ ", " "]:
            with (
                self.subTest(configured=configured),
                patch.dict("os.environ", {"FRAME_STATIC_DIR": configured}),
                patch("app.data_provider.TRANSCRIPT_CHUNK_SEARCH_ENABLED", True),
                patch("app.data_provider.PECoreTextEncoder"),
                patch.object(DataProvider, "_get_collection"),
            ):
                provider = DataProvider()
                self.addCleanup(provider.close)
                self.assertEqual(provider.transcript_search_service._keyframe_dir, main._frame_static_dir())

    async def check_startup(self, shared):
        provider = MagicMock()
        provider.transcript_search_service = shared
        with (
            patch.dict("os.environ", {"WARMUP_TEXT_ENCODER": "false", "WARMUP_TRANSLATION": "false",
                                      "TRANSCRIPT_CHUNK_SEARCH_ENABLED": "true", "WARMUP_TRANSCRIPT_SEARCH": "true"}),
            patch.object(main, "_data_provider", None),
            patch.object(main, "_transcript_search_service", None),
            patch.object(main, "_strategies", {}),
            patch.object(main.postgres_client, "init_schema", new=AsyncMock()),
            patch.object(main.postgres_client, "close_pool", new=AsyncMock()),
            patch.object(main.milvus_client, "connect"),
            patch.object(main.milvus_client, "create_collection_if_missing"),
            patch.object(main.milvus_client, "create_transcript_collection_if_missing"),
            patch.object(main, "DataProvider", return_value=provider),
            patch.object(main, "TranscriptSearchService") as factory,
            patch.object(main, "discover_strategies", return_value={}),
        ):
            async with main.lifespan(main.app):
                expected = shared if shared is not None else factory.return_value
                self.assertIs(main._transcript_search_service, expected)
                expected.warmup.assert_called_once_with()
                if shared is not None:
                    factory.assert_not_called()
                else:
                    factory.assert_called_once_with(keyframe_dir=main.FRAME_STATIC_DIR)
            provider.close.assert_called_once_with()

    async def test_startup_warms_existing_service_without_loading_a_second_model(self):
        await self.check_startup(MagicMock())

    async def test_startup_retains_fallback_when_provider_service_is_unavailable(self):
        await self.check_startup(None)

    async def test_shared_query_cache_returns_identical_vectors_without_reencoding(self):
        service = TranscriptSearchService()
        provider = DataProvider.__new__(DataProvider)
        provider._transcript_search = service
        vector = np.arange(384, dtype=np.float32)
        self.addCleanup(service._cached_encode_query.cache_clear)
        with patch.object(service, "_encode_query_uncached", return_value=vector) as encode:
            first = service.encode_query("geography lesson")
            second = provider.transcript_search_service.encode_query("geography lesson")
        np.testing.assert_array_equal(first, second)
        encode.assert_called_once_with("geography lesson")
