"""Bound synchronous database work without blocking the HTTP event loop."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class BlockingIO:
    def __init__(self, workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="database")
        self._slots = asyncio.Semaphore(workers)

    async def run(self, function, *args, **kwargs):
        await self._slots.acquire()
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self._executor, partial(function, *args, **kwargs)
            )
        except BaseException:
            self._slots.release()
            raise

        def finished(done):
            self._slots.release()
            if not done.cancelled():
                done.exception()

        future.add_done_callback(finished)
        # A disconnected/timed-out caller must not free a slot while its DB call runs.
        return await asyncio.shield(future)

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
