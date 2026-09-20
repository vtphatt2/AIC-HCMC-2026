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

    def __init__(self, directory: str, max_bytes: int):
        self.directory = Path(directory)
        self.shard_budget = max_bytes // self.SHARDS

    def _path(self, key):
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.directory / str(int(digest[:2], 16) % self.SHARDS) / (digest + ".jpg")

    async def read(self, key):
        return await asyncio.to_thread(self._read, self._path(key))

    @staticmethod
    def _read(path):
        try:
            data = path.read_bytes()
            os.utime(path, None)
            return data
        except FileNotFoundError:
            return None

    def _write(self, path, data):
        # A process killed during a write may leave an unfinished temporary file.
        for unfinished in path.parent.glob("*.tmp"):
            unfinished.unlink()
        if len(data) > self.shard_budget:
            return
        entries = sorted(path.parent.glob("*.jpg"), key=lambda p: p.stat().st_mtime_ns)
        sizes = [(p, p.stat().st_size) for p in entries]
        total = sum(size for _, size in sizes)
        for old, size in sizes:
            if total + len(data) <= self.shard_budget:
                break
            old.unlink()
            total -= size
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                handle.write(data)
            os.replace(temporary, path)
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
        # Fixed shard locks avoid unbounded lock files and survive worker restarts.
        with (shard / ".lock").open("a+b") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(.01)
            try:
                data = await asyncio.to_thread(self._read, path)
                if data is None:
                    data = await produce()
                    write = asyncio.create_task(asyncio.to_thread(self._write, path, data))
                    try:
                        await asyncio.shield(write)
                    except asyncio.CancelledError:
                        # Keep the interprocess lock until the write really finishes.
                        await write
                        raise
                    except OSError:
                        logger.warning("Could not cache JPEG; returning decoded bytes", exc_info=True)
                return data
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
