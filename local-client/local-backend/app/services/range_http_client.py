"""Resilient HTTP Range-fetch wrapper — the retry/backoff layer the
organizers' own toolkit's RangeHTTPClient has (see
docs/remote_zip_video_toolkit's http.py) that our earlier plain httpx calls
in zip_frame_source.py / remote_zip_proxy.py didn't.

This is our own proxy's problem to own: every failure funnels through
RangeFetchError so a caller is guaranteed either bytes back or a clear,
typed exception — never a raw network exception, and never a hang past
max_retries worth of bounded per-attempt timeouts.
"""
from __future__ import annotations

import asyncio
import logging
import random

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RangeFetchError(RuntimeError):
    """An upstream Range GET could not be satisfied after retries — the
    caller has no compressed bytes to work with for this request."""


class RangeHTTPClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        max_retries: int = 3,
        timeout_sec: float = 10.0,
        backoff_base: float = 0.5,
    ):
        self._client = client
        self._max_retries = max(1, max_retries)
        self._timeout_sec = timeout_sec
        self._backoff_base = backoff_base

    async def fetch(self, url: str, start: int, size: int) -> bytes:
        """Fetch exactly `size` bytes starting at `start`. Always returns
        those bytes or raises RangeFetchError — never anything else."""
        if size <= 0:
            return b""

        end = start + size - 1
        last_error: str = "unknown error"

        for attempt in range(self._max_retries):
            if attempt > 0:
                delay = self._backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)

            try:
                resp = await self._client.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=self._timeout_sec,
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Range fetch attempt %d/%d failed (%s) for %s bytes=%d-%d",
                    attempt + 1, self._max_retries, last_error, url, start, end,
                )
                continue

            if resp.status_code == 206:
                data = resp.content
                if len(data) == size:
                    return data
                last_error = f"requested {size} bytes but received {len(data)}"
            elif resp.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}"
            else:
                # Not retryable: a 4xx (other than 429) means retrying won't help.
                raise RangeFetchError(
                    f"Unexpected HTTP {resp.status_code} for {url} bytes={start}-{end}"
                )

            logger.warning(
                "Range fetch attempt %d/%d failed (%s) for %s bytes=%d-%d",
                attempt + 1, self._max_retries, last_error, url, start, end,
            )

        raise RangeFetchError(
            f"Range fetch failed after {self._max_retries} attempts for {url} "
            f"bytes={start}-{end}: {last_error}"
        )
