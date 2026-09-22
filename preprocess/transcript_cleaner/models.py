from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TranscriptError(ValueError):
    """The transcript or a model response violates the data contract."""


@dataclass(frozen=True, slots=True)
class Segment:
    index: int
    start_time_ms: int
    end_time_ms: int
    text: str
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: int
    segments: tuple[Segment, ...]
    estimated_input_tokens: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CleanedChunk:
    chunk_id: int
    texts_by_index: dict[int, str]
    attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float | None = None
    fallback_segment_count: int = 0


@dataclass(slots=True)
class FileJob:
    source_path: Path
    output_path: Path
    segments: list[Segment]
    chunks: list[Chunk]
    job_fingerprint: str
    preserved_texts: dict[int, str] = field(default_factory=dict)
    results: dict[int, CleanedChunk] = field(default_factory=dict)
    errors: dict[int, Exception] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkItem:
    job: FileJob
    chunk: Chunk
