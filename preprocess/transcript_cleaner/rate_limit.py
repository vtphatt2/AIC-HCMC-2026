from __future__ import annotations

import asyncio
import time
from collections import deque


class SlidingWindowRateLimiter:
    """A shared, conservative 60-second RPM/input-TPM limiter."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self._rpm = max(0, requests_per_minute)
        self._tpm = max(0, tokens_per_minute)
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_input_tokens: int) -> None:
        if self._rpm == 0 and self._tpm == 0:
            return
        if self._tpm and estimated_input_tokens > self._tpm:
            raise ValueError(
                f"One request ({estimated_input_tokens} estimated tokens) exceeds TPM "
                f"limit ({self._tpm})"
            )

        while True:
            wait_seconds = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._events and self._events[0][0] <= cutoff:
                    self._events.popleft()

                request_blocked = bool(self._rpm and len(self._events) >= self._rpm)
                used_tokens = sum(tokens for _, tokens in self._events)
                token_blocked = bool(
                    self._tpm and used_tokens + estimated_input_tokens > self._tpm
                )
                if not request_blocked and not token_blocked:
                    self._events.append((now, estimated_input_tokens))
                    return

                expirations: list[float] = []
                if request_blocked:
                    expirations.append(self._events[0][0] + 60.0)
                if token_blocked:
                    running = used_tokens
                    for timestamp, tokens in self._events:
                        running -= tokens
                        if running + estimated_input_tokens <= self._tpm:
                            expirations.append(timestamp + 60.0)
                            break
                wait_seconds = max(0.01, min(expirations) - now)
            await asyncio.sleep(wait_seconds)


class SharedCooldown:
    """Pauses all workers after a server asks one worker to back off."""

    def __init__(self) -> None:
        self._until = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        while True:
            delay = self._until - time.monotonic()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def extend(self, seconds: float) -> None:
        async with self._lock:
            self._until = max(self._until, time.monotonic() + max(0.0, seconds))
