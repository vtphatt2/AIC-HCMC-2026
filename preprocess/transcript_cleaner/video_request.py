from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .models import Chunk, Segment


@dataclass(frozen=True, slots=True)
class VideoRequestConfig:
    """Sizing guard for the single request that represents one video."""

    max_estimated_input_tokens: int = 100_000
    prompt_overhead_tokens: int = 350
    characters_per_token: float = 3.0

    def __post_init__(self) -> None:
        if self.max_estimated_input_tokens <= self.prompt_overhead_tokens:
            raise ValueError(
                "max_estimated_input_tokens must exceed prompt_overhead_tokens"
            )
        if self.characters_per_token <= 0:
            raise ValueError("characters_per_token must be positive")


def _estimate_tokens(segments: list[Segment], config: VideoRequestConfig) -> int:
    payload = json.dumps(
        [{"id": segment.index, "text": segment.text} for segment in segments],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return config.prompt_overhead_tokens + max(
        1, math.ceil(len(payload) / config.characters_per_token)
    )


def _fingerprint(segments: list[Segment]) -> str:
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(str(segment.index).encode("ascii"))
        digest.update(b"\0")
        digest.update(segment.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def create_video_request(
    segments: list[Segment], config: VideoRequestConfig
) -> Chunk:
    """Create exactly one request; never silently split a video."""

    if not segments:
        raise ValueError("A Gemini video request must contain at least one segment")
    estimated_tokens = _estimate_tokens(segments, config)
    if estimated_tokens > config.max_estimated_input_tokens:
        raise ValueError(
            f"Video request is estimated at {estimated_tokens} input tokens, above the "
            f"configured limit of {config.max_estimated_input_tokens}; no request was sent"
        )
    return Chunk(
        chunk_id=0,
        segments=tuple(segments),
        estimated_input_tokens=estimated_tokens,
        fingerprint=_fingerprint(segments),
    )
