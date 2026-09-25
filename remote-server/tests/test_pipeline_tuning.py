import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from app.services.blocking_io import BlockingIO
from app.services.jpeg_disk_cache import JpegDiskCache


class BlockingIOTests(unittest.IsolatedAsyncioTestCase):
    async def test_loop_remains_responsive_and_result_is_preserved(self):
        pool = BlockingIO(1)
        self.addCleanup(pool.close)
        started, release = threading.Event(), threading.Event()
        def work():
            started.set()
            release.wait(2)
            return ["same", "ordered", "results"]
        task = asyncio.create_task(pool.run(work))
        while not started.is_set():
            await asyncio.sleep(.001)
        self.assertFalse(task.done())
        release.set()
        self.assertEqual(await task, ["same", "ordered", "results"])

    async def test_cancelled_caller_keeps_slot_until_worker_finishes(self):
        pool = BlockingIO(1)
        self.addCleanup(pool.close)
        started, release, second_started = threading.Event(), threading.Event(), threading.Event()
        def first():
            started.set()
            release.wait(2)
        first_task = asyncio.create_task(pool.run(first))
        while not started.is_set():
            await asyncio.sleep(.001)
        first_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_task
        second = asyncio.create_task(pool.run(second_started.set))
        await asyncio.sleep(.03)
        self.assertFalse(second_started.is_set())
        release.set()
        await second
        self.assertTrue(second_started.is_set())

    async def test_failure_releases_slot(self):
        pool = BlockingIO(1)
        self.addCleanup(pool.close)
        def fail():
            raise ValueError("database failure")
        with self.assertRaisesRegex(ValueError, "database failure"):
            await pool.run(fail)
        self.assertEqual(await pool.run(lambda: 42), 42)


class DiskCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_presence_check_does_not_read_or_touch_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JpegDiskCache(directory, 4096)
            async def produce():
                return b"jpeg"
            self.assertFalse(await cache.contains("key"))
            await cache.get("key", produce)
            path = cache._path("key")
            before = path.stat().st_mtime_ns
            with patch.object(cache, "_read", side_effect=AssertionError("JPEG should not be read")):
                self.assertTrue(await cache.contains("key"))
            self.assertEqual(path.stat().st_mtime_ns, before)

    async def test_write_failure_returns_already_decoded_image(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JpegDiskCache(directory, 4096)
            async def produce():
                return b"jpeg"
            with patch.object(cache, "_write", side_effect=OSError("disk full")):
                with self.assertLogs("app.services.jpeg_disk_cache", level="WARNING"):
                    self.assertEqual(await cache.get("key", produce), b"jpeg")

    async def test_cancelled_write_holds_lock_until_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JpegDiskCache(directory, 4096)
            started, release = threading.Event(), threading.Event()
            original = cache._write
            calls = []
            def slow_write(path, data):
                started.set()
                release.wait(2)
                original(path, data)
            async def produce():
                calls.append(1)
                return b"jpeg"
            with patch.object(cache, "_write", side_effect=slow_write):
                first = asyncio.create_task(cache.get("key", produce))
                while not started.is_set():
                    await asyncio.sleep(.001)
                first.cancel()
                second = asyncio.create_task(JpegDiskCache(directory, 4096).get("key", produce))
                await asyncio.sleep(.03)
                self.assertFalse(second.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertEqual(await second, b"jpeg")
                self.assertEqual(calls, [1])

    async def test_shared_bytes_and_versioned_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = JpegDiskCache(directory, 4096), JpegDiskCache(directory, 4096)
            calls = []
            async def produce():
                calls.append(1)
                await asyncio.sleep(.02)
                return b"identical JPEG bytes"
            self.assertEqual(await asyncio.gather(a.get("v1", produce), b.get("v1", produce)),
                             [b"identical JPEG bytes"] * 2)
            self.assertEqual(len(calls), 1)
            await b.get("v2", produce)
            self.assertEqual(len(calls), 2)

    async def test_failure_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JpegDiskCache(directory, 4096)
            async def fail():
                raise RuntimeError("decode failed")
            with self.assertRaises(RuntimeError):
                await cache.get("key", fail)
            self.assertEqual(list(Path(directory).rglob("*.jpg")), [])

    async def test_budget_and_oversized_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JpegDiskCache(directory, 4096)
            async def produce():
                return b"x" * 64
            for i in range(100):
                await cache.get(str(i), produce)
            self.assertLessEqual(sum(p.stat().st_size for p in Path(directory).rglob("*.jpg")), 4096)
            async def oversized():
                return b"x" * 65
            self.assertEqual(len(await cache.get("oversized", oversized)), 65)
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])

    async def test_two_processes_decode_once(self):
        with tempfile.TemporaryDirectory() as directory:
            script = '''import asyncio,sys
from pathlib import Path
from app.services.jpeg_disk_cache import JpegDiskCache
async def produce():
    with (Path(sys.argv[1])/"calls").open("ab") as f: f.write(b"1")
    await asyncio.sleep(.15)
    return b"jpeg"
async def main():
    assert await JpegDiskCache(sys.argv[1],4096).get("same-frame",produce)==b"jpeg"
asyncio.run(main())
'''
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
            processes = [subprocess.Popen([sys.executable, "-c", script, directory], env=env) for _ in range(2)]
            for process in processes:
                self.assertEqual(await asyncio.to_thread(process.wait, 5), 0)
            self.assertEqual((Path(directory) / "calls").read_bytes(), b"1")
