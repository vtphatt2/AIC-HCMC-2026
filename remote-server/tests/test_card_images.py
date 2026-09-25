import ast
import asyncio
from collections import OrderedDict
import io
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image, ImageDraw

from app.services import local_zip_media as media
from app.services.jpeg_disk_cache import JpegDiskCache


def jpeg(size=(1280, 720)):
    image = Image.new("RGB", size, "navy")
    ImageDraw.Draw(image).rectangle((0, 0, size[0] // 2, size[1]), fill="orange")
    out = io.BytesIO()
    image.save(out, "JPEG", quality=95)
    return out.getvalue()


class CardImageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        source = Path(self.tmp.name) / "source.zip"
        source.write_bytes(b"source identity")
        index = SimpleNamespace(zip_path=source, data_offset=100, fps=25.0)
        self.original = jpeg()
        self.decoder = AsyncMock(return_value=self.original)
        for name, value in {
            "_jpeg_cache": OrderedDict(), "_jpeg_cache_bytes": 0,
            "_jpeg_inflight": {}, "_jpeg_waiters": {},
            "_card_activity_task": None,
            "JPEG_CACHE_MAX_BYTES": 1024 * 1024,
            "_disk_cache": None, "_card_disk_cache": None,
            "_decode_semaphore": asyncio.Semaphore(3),
            "_resize_semaphore": asyncio.Semaphore(2),
            "_get_index": AsyncMock(return_value=index),
            "_decode_frame_jpeg": self.decoder,
        }.items():
            patcher = patch.object(media, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_original_bytes_unchanged_and_thumbnail_is_same_frame(self):
        full, card = await asyncio.gather(
            media.get_frame_jpeg("V1", 100), media.get_frame_jpeg("V1", 100, width=640))
        self.assertEqual(full, self.original)
        self.assertEqual(self.decoder.await_count, 1)
        self.assertLess(len(card), len(full))
        with Image.open(io.BytesIO(card)) as image:
            self.assertEqual(image.size, (640, 360))
            self.assertGreater(image.getpixel((50, 50))[0], 240)
            self.assertLess(image.getpixel((590, 50))[0], 10)
        self.assertEqual(await media.get_frame_jpeg("V1", 100), full)

    async def test_small_source_is_not_upscaled_or_reencoded(self):
        self.decoder.return_value = jpeg((320, 180))
        self.assertEqual(await media.get_frame_jpeg("V1", 100, width=640), self.decoder.return_value)

    async def test_simultaneous_sizes_decode_once_and_share_thumbnail(self):
        with patch.object(media, "_resize_jpeg", wraps=media._resize_jpeg) as resize:
            results = await asyncio.gather(*[
                media.get_frame_jpeg("V1", 100, width=640) for _ in range(12)])
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(self.decoder.await_count, 1)
        self.assertEqual(resize.call_count, 1)

    async def test_ready_card_does_not_wait_behind_unrelated_video_decodes(self):
        await media.get_frame_jpeg("V1", 100)
        with patch.object(media, "_decode_semaphore", asyncio.Semaphore(1)):
            async with media._decode_semaphore:
                card = await asyncio.wait_for(media.get_frame_jpeg("V1", 100, width=640), 1)
        self.assertEqual(Image.open(io.BytesIO(card)).size, (640, 360))

    async def test_disk_variants_survive_restart_without_nested_shard_deadlock(self):
        cache = JpegDiskCache(str(Path(self.tmp.name) / "cache"), 1024 * 1024)
        # Deliberately collide every original/thumbnail key.
        cache.SHARDS = 1
        cache.shard_budget = 1024 * 1024
        with patch.object(media, "_disk_cache", cache):
            card = await asyncio.wait_for(media.get_frame_jpeg("V1", 100, width=640), 2)
            media._jpeg_cache.clear()
            media._jpeg_cache_bytes = 0
            self.decoder.side_effect = AssertionError("A warm disk variant must not decode")
            with patch.object(media, "_resize_jpeg", side_effect=AssertionError("Must reuse card")):
                self.assertEqual(await media.get_frame_jpeg("V1", 100, width=640), card)
                self.assertEqual(await media.get_frame_jpeg("V1", 100), self.original)

    async def test_unrelated_cache_keys_do_not_wait_for_a_slow_decode(self):
        cache = JpegDiskCache(str(Path(self.tmp.name) / "collision-cache"), 1024 * 1024)
        cache.SHARDS = 1
        cache.shard_budget = 1024 * 1024
        first, second = "first", "second"
        while cache._path(first).stem[:3] == cache._path(second).stem[:3]:
            second += "x"
        started, release = asyncio.Event(), asyncio.Event()

        async def slow():
            started.set()
            await release.wait()
            return b"first jpeg"

        async def fast():
            return b"second jpeg"

        waiting = asyncio.create_task(cache.get(first, slow))
        await started.wait()
        # Both keys share the same write shard. A slow producer must not hold
        # its lock, because a live result card may need the other key.
        self.assertEqual(await asyncio.wait_for(cache.get(second, fast), .5), b"second jpeg")
        release.set()
        self.assertEqual(await waiting, b"first jpeg")

    async def test_same_cache_key_still_produces_only_once(self):
        cache = JpegDiskCache(str(Path(self.tmp.name) / "single-flight-cache"), 1024 * 1024)
        calls = 0

        async def produce():
            nonlocal calls
            calls += 1
            await asyncio.sleep(.02)
            return b"shared jpeg"

        values = await asyncio.gather(*(cache.get("same-key", produce) for _ in range(8)))
        self.assertEqual(values, [b"shared jpeg"] * 8)
        self.assertEqual(calls, 1)

    async def test_background_warmer_yields_while_live_decode_is_active(self):
        cache = JpegDiskCache(str(Path(self.tmp.name) / "live-cards"), 1024 * 1024)
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_decode(*args):
            started.set()
            await release.wait()
            return self.original

        self.decoder.side_effect = slow_decode
        with patch.object(media, "_card_disk_cache", cache):
            live = asyncio.create_task(media.get_frame_jpeg("V1", 100, width=640))
            await started.wait()
            marker = media.card_activity_file()
            for _ in range(50):
                if marker.exists():
                    break
                await asyncio.sleep(.01)
            self.assertTrue(marker.exists())
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(media.wait_for_live_card_idle(.5), .65)
            release.set()
            self.assertTrue(await live)
            await asyncio.wait_for(media.wait_for_live_card_idle(.5), 1.5)

    async def test_separate_card_cache_keeps_original_bytes_and_reuses_card(self):
        original_cache = JpegDiskCache(str(Path(self.tmp.name) / "originals"), 16 * 1024 * 1024)
        card_cache = JpegDiskCache(str(Path(self.tmp.name) / "cards"), 16 * 1024 * 1024)
        with patch.object(media, "_disk_cache", original_cache), patch.object(media, "_card_disk_cache", card_cache):
            full, card = await asyncio.gather(
                media.get_frame_jpeg("V1", 100), media.get_frame_jpeg("V1", 100, width=640))
            self.assertEqual(full, self.original)
            self.assertEqual(self.decoder.await_count, 1)
            self.assertEqual(len(list(card_cache.directory.rglob("*.jpg"))), 1)
            self.assertEqual(len(list(original_cache.directory.rglob("*.jpg"))), 1)
            media._jpeg_cache.clear()
            media._jpeg_cache_bytes = 0
            self.decoder.side_effect = AssertionError("Warm variants must not decode")
            self.assertEqual(await media.get_frame_jpeg("V1", 100, width=640), card)
            self.assertEqual(await media.get_frame_jpeg("V1", 100), full)

    async def test_cancelled_card_viewer_does_not_discard_shared_work(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def decode(*args):
            started.set()
            await release.wait()
            return self.original

        self.decoder.side_effect = decode
        first = asyncio.create_task(media.get_frame_jpeg("V1", 100, width=640))
        await started.wait()
        second = asyncio.create_task(media.get_frame_jpeg("V1", 100, width=640))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        release.set()
        self.assertTrue(await second)
        self.assertEqual(self.decoder.await_count, 1)

    async def test_abandoned_card_stops_decoding_and_can_be_retried(self):
        started = asyncio.Event()

        async def decode(*args):
            started.set()
            await asyncio.Future()

        self.decoder.side_effect = decode
        task = asyncio.create_task(media.get_frame_jpeg("V1", 100, width=640))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        self.assertFalse(media._jpeg_inflight)
        self.assertFalse(media._jpeg_waiters)
        self.decoder.side_effect = None
        self.assertTrue(await media.get_frame_jpeg("V1", 100, width=640))

    async def test_disk_read_failure_still_returns_card(self):
        cache = JpegDiskCache(str(Path(self.tmp.name) / "cache"), 1024 * 1024)
        with patch.object(media, "_disk_cache", cache), patch.object(cache, "read", side_effect=OSError("unavailable")):
            with self.assertLogs(media.logger, level="WARNING"):
                card = await media.get_frame_jpeg("V1", 100, width=640)
        self.assertEqual(Image.open(io.BytesIO(card)).size, (640, 360))

    async def test_webp_card_shares_original_but_has_separate_cached_bytes(self):
        full, jpg, webp, again = await asyncio.gather(
            media.get_frame_jpeg("V1", 100),
            media.get_frame_jpeg("V1", 100, width=640),
            media.get_frame_jpeg("V1", 100, width=640, format="webp"),
            media.get_frame_jpeg("V1", 100, width=640, format="webp"))
        self.assertEqual(full, self.original)
        self.assertEqual(webp, again)
        self.assertEqual(self.decoder.await_count, 1)
        self.assertEqual(Image.open(io.BytesIO(jpg)).format, "JPEG")
        with Image.open(io.BytesIO(webp)) as image:
            self.assertEqual(image.format, "WEBP")
            self.assertEqual(image.size, (640, 360))
            self.assertGreater(image.getpixel((50, 50))[0], 240)
            self.assertLess(image.getpixel((590, 50))[0], 15)
        self.assertEqual(await media.get_frame_jpeg("V1", 100, width=640), jpg)

    async def test_webp_disk_cache_is_distinct_and_reusable(self):
        cache = JpegDiskCache(str(Path(self.tmp.name) / "cache"), 1024 * 1024)
        cache.SHARDS = 1
        cache.shard_budget = 1024 * 1024
        with patch.object(media, "_disk_cache", cache):
            jpg = await media.get_frame_jpeg("V1", 100, width=640)
            webp = await asyncio.wait_for(media.get_frame_jpeg("V1", 100, width=640, format="webp"), 2)
            media._jpeg_cache.clear()
            media._jpeg_cache_bytes = 0
            self.decoder.side_effect = AssertionError("Must reuse cached variants")
            self.assertEqual(await media.get_frame_jpeg("V1", 100, width=640, format="webp"), webp)
            self.assertEqual(await media.get_frame_jpeg("V1", 100, width=640), jpg)

    async def test_webp_does_not_upscale_or_replace_original(self):
        self.decoder.return_value = jpeg((320, 180))
        webp = await media.get_frame_jpeg("V1", 100, width=640, format="webp")
        self.assertEqual(Image.open(io.BytesIO(webp)).size, (320, 180))
        self.assertEqual(await media.get_frame_jpeg("V1", 100), self.decoder.return_value)
        with self.assertRaises(ValueError):
            await media.get_frame_jpeg("V1", 100, format="webp")

    async def test_http_original_variant_and_restricted_dimensions(self):
        import httpx
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.responses import Response

        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "zip_frame"]
        app = FastAPI()
        scope = dict(app=app, asyncio=asyncio, Query=Query, Request=Request,
                     HTTPException=HTTPException,
                     Response=Response, local_zip_media=media, ZIP_FRAME_TIMEOUT_SEC=2,
                     logger=logging.getLogger(__name__), _require_released_video=lambda _: None)
        exec(compile(tree, "main.py", "exec"), scope)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            full = await client.get("/api/zip-frame/V1/100")
            card = await client.get("/api/zip-frame/V1/100?width=640")
            self.assertEqual(full.content, self.original)
            self.assertEqual(card.status_code, 200)
            self.assertEqual(Image.open(io.BytesIO(card.content)).size, (640, 360))
            self.assertIn("immutable", card.headers["cache-control"])
            webp = await client.get("/api/zip-frame/V1/100?width=640&format=webp")
            self.assertEqual(webp.status_code, 200)
            self.assertEqual(webp.headers["content-type"], "image/webp")
            self.assertEqual(Image.open(io.BytesIO(webp.content)).format, "WEBP")
            self.assertIn("immutable", webp.headers["cache-control"])
            for url in ["?format=webp", "?width=640&format=png"]:
                self.assertEqual((await client.get("/api/zip-frame/V1/100" + url)).status_code, 422)
            for width in ["0", "-1", "641", "999999", "bad"]:
                self.assertEqual((await client.get(f"/api/zip-frame/V1/100?width={width}")).status_code, 422)

    async def test_http_disconnect_cancels_unneeded_decode(self):
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.responses import Response

        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "zip_frame"]
        scope = dict(app=FastAPI(), asyncio=asyncio, Query=Query, Request=Request,
                     HTTPException=HTTPException, Response=Response, local_zip_media=media,
                     ZIP_FRAME_TIMEOUT_SEC=2, logger=logging.getLogger(__name__),
                     _require_released_video=lambda _: None)
        exec(compile(tree, "main.py", "exec"), scope)

        started, cancelled = asyncio.Event(), asyncio.Event()

        async def decode(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        class Gone:
            async def is_disconnected(self):
                return True

        with patch.object(media, "get_frame_jpeg", side_effect=decode):
            response = await scope["zip_frame"]("V1", 100, Gone(), width=640, format="jpeg", frame_number=None)
        self.assertEqual(response.status_code, 204)
        self.assertTrue(started.is_set())
        self.assertTrue(cancelled.is_set())

    async def test_cancelled_ffmpeg_job_reaps_its_process(self):
        pid_file = Path(self.tmp.name) / "decoder.pid"
        command = [sys.executable, "-c",
                   "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(30)",
                   str(pid_file)]
        task = asyncio.create_task(media._run_ffmpeg(command, timeout_sec=20))
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(.01)
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
