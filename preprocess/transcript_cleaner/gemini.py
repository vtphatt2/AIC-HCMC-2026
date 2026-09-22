from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

from .models import Chunk, CleanedChunk, TranscriptError
from .prompt import (
    SYSTEM_INSTRUCTION_MARKER,
    SYSTEM_INSTRUCTION_SEGMENTS,
    segment_marker,
    user_prompt_marker,
    user_prompt_segments,
)
from .rate_limit import SharedCooldown, SlidingWindowRateLimiter


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class GeminiAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    model: str = "gemini-3.5-flash-lite"
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    backoff_base_seconds: float = 1.0
    backoff_cap_seconds: float = 60.0
    temperature: float = 0.1
    max_output_tokens: int = 65_536
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    payload_format: str = "json-segments"
    max_server_retry_after_seconds: float = 120.0
    max_validation_attempts: int = 2

    def __post_init__(self) -> None:
        if not self.model or "/" in self.model or ".." in self.model:
            raise ValueError("Invalid Gemini model name")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.max_validation_attempts < 1:
            raise ValueError("max_validation_attempts must be positive")
        if self.payload_format not in {"json-segments", "marker-text"}:
            raise ValueError("payload_format must be json-segments or marker-text")


class GeminiCleaner:
    def __init__(
        self,
        *,
        api_key: str,
        config: GeminiConfig,
        rate_limiter: SlidingWindowRateLimiter,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key is empty")
        self._config = config
        self._limiter = rate_limiter
        self._cooldown = SharedCooldown()
        self._abort_event = asyncio.Event()
        timeout = httpx.Timeout(config.timeout_seconds, connect=min(15.0, config.timeout_seconds))
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
        )

    async def __aenter__(self) -> GeminiCleaner:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def abort(self) -> None:
        """Prevent pending/retrying work from starting another HTTP attempt."""

        self._abort_event.set()

    async def clean(self, chunk: Chunk) -> CleanedChunk:
        last_error: Exception | None = None
        validation_failures = 0
        last_attempt = 0
        started = time.monotonic()
        for attempt in range(1, self._config.max_attempts + 1):
            last_attempt = attempt
            if self._abort_event.is_set():
                raise GeminiAPIError("Batch stopped before another API attempt", 429)
            try:
                await self._cooldown.wait()
                await self._limiter.acquire(chunk.estimated_input_tokens)
                if self._abort_event.is_set():
                    raise GeminiAPIError("Batch stopped before another API attempt", 429)
                response = await self._request(chunk)
                texts, input_tokens, output_tokens, fallback_count = self._parse_response(
                    response, chunk
                )
                return CleanedChunk(
                    chunk_id=chunk.chunk_id,
                    texts_by_index=texts,
                    attempts=attempt,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=time.monotonic() - started,
                    fallback_segment_count=fallback_count,
                )
            except TranscriptError as exc:
                last_error = exc
                retry_after = None
                validation_failures += 1
                if validation_failures >= self._config.max_validation_attempts:
                    break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                retry_after = None
            except GeminiAPIError as exc:
                last_error = exc
                if exc.status_code not in RETRYABLE_STATUS_CODES:
                    raise
                retry_after = getattr(exc, "retry_after", None)

            if (
                retry_after is not None
                and retry_after > self._config.max_server_retry_after_seconds
            ):
                break

            if attempt >= self._config.max_attempts:
                break
            exponential = min(
                self._config.backoff_cap_seconds,
                self._config.backoff_base_seconds * (2 ** (attempt - 1)),
            )
            delay = random.uniform(0.0, exponential)
            if retry_after is not None:
                delay = max(delay, retry_after)
            if isinstance(last_error, GeminiAPIError) and last_error.status_code == 429:
                await self._cooldown.extend(delay)
            await asyncio.sleep(delay)

        final_status = (
            last_error.status_code if isinstance(last_error, GeminiAPIError) else None
        )
        raise GeminiAPIError(
            f"Chunk {chunk.chunk_id} failed after {last_attempt} attempts: "
            f"{last_error}",
            status_code=final_status,
        ) from last_error

    async def _request(self, chunk: Chunk) -> dict[str, Any]:
        if self._config.payload_format == "marker-text":
            transcript = "\n".join(
                segment_marker(segment.index) + segment.text
                for segment in chunk.segments
            )
            system_instruction = SYSTEM_INSTRUCTION_MARKER
            request_text = user_prompt_marker(transcript)
            response_schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {"transcript": {"type": "string"}},
                "required": ["transcript"],
            }
        else:
            payload = [
                {"id": segment.index, "text": segment.text}
                for segment in chunk.segments
            ]
            system_instruction = SYSTEM_INSTRUCTION_SEGMENTS
            request_text = user_prompt_segments(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            response_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                },
            }
            if len(chunk.segments) <= 128:
                response_schema["minItems"] = len(chunk.segments)
                response_schema["maxItems"] = len(chunk.segments)
        body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": request_text}],
                }
            ],
            "generationConfig": {
                "temperature": self._config.temperature,
                "maxOutputTokens": self._config.max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            },
        }
        model = quote(self._config.model, safe="-_.")
        url = f"{self._config.base_url}/models/{model}:generateContent"
        response = await self._client.post(url, json=body)
        if response.status_code >= 400:
            message = _safe_error_message(response)
            error = GeminiAPIError(message, response.status_code)
            error.retry_after = _retry_after_seconds(response)  # type: ignore[attr-defined]
            raise error
        try:
            result = response.json()
        except ValueError as exc:
            raise TranscriptError("Gemini returned a non-JSON API response") from exc
        if not isinstance(result, dict):
            raise TranscriptError("Gemini API response must be an object")
        return result

    def _parse_response(
        self,
        response: dict[str, Any], chunk: Chunk
    ) -> tuple[dict[int, str], int | None, int | None, int]:
        try:
            candidate = response["candidates"][0]
            finish_reason = candidate.get("finishReason")
            if finish_reason != "STOP":
                raise TranscriptError(f"Unexpected finish reason: {finish_reason}")
            parts = candidate["content"]["parts"]
            response_text = "".join(
                part.get("text", "")
                for part in parts
                if not part.get("thought", False)
            )
            structured = json.loads(response_text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise TranscriptError("Gemini response has no valid structured content") from exc

        if self._config.payload_format == "marker-text":
            texts = self._parse_marker_text(structured, chunk)
            structural_fallback_count = 0
        else:
            texts, structural_fallback_count = self._parse_segment_items(
                structured, chunk
            )

        fallback_count = structural_fallback_count
        fallback_count += self._validate_and_fallback_texts(texts, chunk)
        usage = response.get("usageMetadata") or {}
        input_tokens = _optional_int(usage.get("promptTokenCount"))
        output_tokens = _optional_int(usage.get("candidatesTokenCount"))
        return texts, input_tokens, output_tokens, fallback_count

    @staticmethod
    def _parse_segment_items(
        items: Any, chunk: Chunk
    ) -> tuple[dict[int, str], int]:
        if not isinstance(items, list):
            raise TranscriptError("Gemini structured response must be an array")

        expected_texts = {
            segment.index: segment.text.strip() for segment in chunk.segments
        }
        expected = set(expected_texts)
        texts: dict[int, str] = {}
        seen_expected_ids: set[int] = set()
        invalid_ids: set[int] = set()
        unexpected_items = 0

        for item in items:
            if not isinstance(item, dict):
                unexpected_items += 1
                continue

            item_id = item.get("id")
            if isinstance(item_id, bool) or not isinstance(item_id, int):
                unexpected_items += 1
                continue
            if item_id not in expected:
                unexpected_items += 1
                continue
            if item_id in seen_expected_ids:
                # Never choose between duplicate values: the source text is the
                # only unambiguous value for this segment.
                unexpected_items += 1
                invalid_ids.add(item_id)
                texts.pop(item_id, None)
                continue

            seen_expected_ids.add(item_id)
            text = item.get("text")
            if set(item) != {"id", "text"}:
                invalid_ids.add(item_id)
                continue
            if not isinstance(text, str):
                invalid_ids.add(item_id)
                continue
            texts[item_id] = text.strip()

        fallback_ids = expected.difference(texts).union(invalid_ids)
        safe_fallback_limit = max(2, math.ceil(len(expected) * 0.05))
        if (
            len(fallback_ids) > safe_fallback_limit
            or unexpected_items > safe_fallback_limit
        ):
            raise TranscriptError(
                "Gemini structural response exceeds safe fallback limit: "
                f"fallback_ids={sorted(fallback_ids)[:10]}, "
                f"unexpected_items={unexpected_items}, "
                f"limit={safe_fallback_limit}"
            )

        for item_id in fallback_ids:
            texts[item_id] = expected_texts[item_id]
        return texts, len(fallback_ids)

    @staticmethod
    def _parse_marker_text(structured: Any, chunk: Chunk) -> dict[int, str]:
        if not isinstance(structured, dict) or set(structured) != {"transcript"}:
            raise TranscriptError("Marker response must contain only transcript")
        transcript = structured["transcript"]
        if not isinstance(transcript, str):
            raise TranscriptError("Marker transcript must be a string")

        marker_pattern = re.compile(r"⟦SEG_(\d{6})⟧")
        matches = list(marker_pattern.finditer(transcript))
        actual_ids = [int(match.group(1)) for match in matches]
        expected_ids = [segment.index for segment in chunk.segments]
        if actual_ids != expected_ids:
            raise TranscriptError(
                "Gemini changed marker sequence: "
                f"expected={expected_ids[:10]}..., actual={actual_ids[:10]}..."
            )
        if matches and transcript[: matches[0].start()].strip():
            raise TranscriptError("Gemini added content before the first marker")

        texts: dict[int, str] = {}
        for position, match in enumerate(matches):
            end = matches[position + 1].start() if position + 1 < len(matches) else len(transcript)
            text = transcript[match.end() : end].strip()
            item_id = int(match.group(1))
            if "⟦SEG_" in text:
                raise TranscriptError(f"Gemini returned a malformed marker near id {item_id}")
            texts[item_id] = text
        return texts

    @staticmethod
    def _validate_and_fallback_texts(
        texts: dict[int, str], chunk: Chunk
    ) -> int:
        expected = {segment.index for segment in chunk.segments}
        if set(texts) != expected:
            raise TranscriptError("Validated response does not cover every segment")

        fallback_count = 0
        for segment in chunk.segments:
            cleaned = texts[segment.index]
            input_length = len(segment.text.strip())
            output_length = len(cleaned)
            unsafe = not cleaned
            unsafe = unsafe or (
                input_length >= 20 and output_length < input_length * 0.60
            )
            unsafe = unsafe or (
                input_length >= 8
                and output_length > max(input_length * 1.80, input_length + 15)
            )
            unsafe = unsafe or (
                _meaningful_numbers(segment.text) != _meaningful_numbers(cleaned)
            )
            unsafe = unsafe or (
                bool(re.fullmatch(r"\s*\[[^\[\]\n]+\]\s*", segment.text))
                and cleaned != segment.text.strip()
            )
            if unsafe:
                texts[segment.index] = segment.text.strip()
                fallback_count += 1

        input_characters = sum(len(segment.text) for segment in chunk.segments)
        output_characters = sum(len(text) for text in texts.values())
        if input_characters and output_characters < input_characters * 0.45:
            raise TranscriptError("Gemini removed an implausibly large amount of content")
        if output_characters > max(200, input_characters * 2.0):
            raise TranscriptError("Gemini added an implausibly large amount of content")
        return fallback_count


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _meaningful_numbers(text: str) -> list[str]:
    """Return standalone numeric tokens, ignoring ASR junk such as ``h5m``."""

    numbers: list[str] = []
    for match in re.finditer(r"\d+(?:[.,]\d+)*", text):
        previous = text[match.start() - 1] if match.start() else ""
        following = text[match.end()] if match.end() < len(text) else ""
        if previous.isalpha() or following.isalpha():
            continue
        numbers.append(match.group(0))
    return numbers


def _safe_error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        message = body.get("error", {}).get("message")
        if isinstance(message, str):
            return f"Gemini HTTP {response.status_code}: {message[:500]}"
    except ValueError:
        pass
    return f"Gemini HTTP {response.status_code}: {response.text[:500]}"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                return max(0.0, when.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass

    try:
        error = response.json().get("error", {})
        for detail in error.get("details", []):
            if not isinstance(detail, dict):
                continue
            retry_delay = detail.get("retryDelay")
            seconds = _duration_seconds(retry_delay)
            if seconds is not None:
                return seconds
        message = error.get("message", "")
        match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.I)
        if match:
            return float(match.group(1))
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def _duration_seconds(value: Any) -> float | None:
    if isinstance(value, str):
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", value.strip())
        return float(match.group(1)) if match else None
    if isinstance(value, dict):
        seconds = value.get("seconds", 0)
        nanos = value.get("nanos", 0)
        if isinstance(seconds, (int, str)) and isinstance(nanos, int):
            try:
                return float(seconds) + nanos / 1_000_000_000
            except ValueError:
                return None
    return None
