"""Bounded, disposable JPEG reuse across Linux backend workers."""
import asyncio
import fcntl
import hashlib
import logging
import os
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class JpegDiskCache:
    SHARDS = 64
    DECODE_LOCKS = 4096

    def __init__(self, directory: str, max_bytes: int):
        self.directory = Path(directory)
        self.shard_budget = max_bytes // self.SHARDS

    def _path(self, key):
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.directory / str(int(digest[:2], 16) % self.SHARDS) / (digest + ".jpg")

    async def read(self, key):
        return await asyncio.to_thread(self._read, self._path(key))

    async def contains(self, key):
        """Check a completed entry without reading the JPEG during a warm-up scan."""
        return await asyncio.to_thread(self._contains, self._path(key))

    @staticmethod
    def _contains(path):
        try:
            return path.stat().st_size > 0
        except FileNotFoundError:
            return False

    @staticmethod
    def _read(path):
        try:
            data = path.read_bytes()
            os.utime(path, None)
            return data
        except FileNotFoundError:
            return None

    def _write(self, path, data):
        if len(data) > self.shard_budget:
            return
        # get() holds the shard's interprocess lock. Track its byte count so a
        # large persistent cache does not stat and sort thousands of JPEGs for
        # every insertion. A missing/bad counter is rebuilt from the files.
        usage_path = path.parent / ".bytes"
        try:
            total = int(usage_path.read_text())
            if total < 0:
                raise ValueError("negative cache size")
        except (OSError, ValueError):
            for unfinished in path.parent.glob("*.tmp"):
                unfinished.unlink()
            total = sum(p.stat().st_size for p in path.parent.glob("*.jpg"))
        if total + len(data) > self.shard_budget:
            entries = sorted(path.parent.glob("*.jpg"), key=lambda p: p.stat().st_mtime_ns)
            for old in entries:
                if total + len(data) <= self.shard_budget:
                    break
                size = old.stat().st_size
                old.unlink()
                total -= size
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                handle.write(data)
            os.replace(temporary, path)
            usage_path.write_text(str(total + len(data)))
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    async def get(self, key: str, produce):
        path = self._path(key)
        shard = path.parent
        await asyncio.to_thread(shard.mkdir, parents=True, exist_ok=True)
        data = await asyncio.to_thread(self._read, path)
        if data is not None:
            return data

        # A decode can take seconds. Holding the cache shard's write lock for
        # that whole time makes unrelated keys in the same shard wait for it.
        # Striped decode locks retain cross-process single-flight for almost
        # all keys while bounding the number of lock files to 4096.
        decode_dir = self.directory / ".decode-locks"
        await asyncio.to_thread(decode_dir.mkdir, parents=True, exist_ok=True)
        decode_stripe = int(path.stem[:3], 16) % self.DECODE_LOCKS
        with (decode_dir / f"{decode_stripe:03x}.lock").open("a+b") as decode_lock:
            await self._lock(decode_lock)
            try:
                data = await asyncio.to_thread(self._read, path)
                if data is not None:
                    return data
                data = await produce()
                # Protect the shard byte counter and eviction only while the
                # completed JPEG is committed, not while FFmpeg is decoding.
                with (shard / ".lock").open("a+b") as write_lock:
                    await self._lock(write_lock)
                    existing = await asyncio.to_thread(self._read, path)
                    if existing is not None:
                        return existing
                    write = asyncio.create_task(asyncio.to_thread(self._write, path, data))
                    try:
                        await asyncio.shield(write)
                    except asyncio.CancelledError:
                        # The writer must finish before either lock is released.
                        await write
                        raise
                    except OSError:
                        logger.warning("Could not cache JPEG; returning decoded bytes", exc_info=True)
                return data
            finally:
                fcntl.flock(decode_lock, fcntl.LOCK_UN)

    @staticmethod
    async def _lock(lock):
        """Wait for an advisory lock without blocking the event loop."""
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                await asyncio.sleep(.01)
