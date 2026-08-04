"""Build non-overlapping, time-aligned sentences from transcript fragments.

This module deliberately sits beside ``transcript_index``: that module remains
responsible for parsing and caching the source transcript fragments, while this
one derives a sentence-level index suitable for mapping a keyframe timestamp to
one complete transcript sentence.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class TranscriptSegment:
    """One timestamped source fragment parsed without the application backend."""

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    """Minimal self-contained transcript contract used by preprocessing."""

    video_id: str
    segments: list[TranscriptSegment]

    @classmethod
    def from_txt(cls, path, video_id: str | None = None) -> "Transcript":
        line_re = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?\]\s*(.*)")
        parsed: list[tuple[int, str, str | None]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            match = line_re.match(line.strip())
            if match is None:
                continue
            hours, minutes, seconds, milliseconds, text = match.groups()
            fraction = (milliseconds or "0").ljust(3, "0")
            start_ms = ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(fraction)
            speaker = "speaker" if text.startswith(">>") else None
            text = text[2:].strip() if speaker else text.strip()
            if text:
                parsed.append((start_ms, text, speaker))
        segments = [
            TranscriptSegment(
                start_ms=start,
                end_ms=max(next_start, start + 1) if next_start is not None else start + 5000,
                text=text,
                speaker=speaker,
            )
            for index, (start, text, speaker) in enumerate(parsed)
            for next_start in [parsed[index + 1][0] if index + 1 < len(parsed) else None]
        ]
        return cls(video_id or path.stem.removesuffix("_Transcript"), segments)


_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’»]*(?=\s|$)")
# ASR transcripts sometimes contain no terminal punctuation for an entire
# video. In that case, split only at original fragment boundaries so text and
# time ownership remain disjoint.
FALLBACK_MAX_DURATION_MS = 15_000
FALLBACK_MAX_CHARACTERS = 320


@dataclass(frozen=True)
class TranscriptSentence:
    """One half-open time interval containing exactly one sentence."""

    sentence_id: str
    start_ms: int
    end_ms: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _SourceSpan:
    char_start: int
    char_end: int
    start_ms: int
    end_ms: int


def build_sentences(transcript: Transcript) -> list[TranscriptSentence]:
    """Return complete sentences with non-overlapping text and time ranges."""
    full_text, spans = _join_segments(transcript.segments)
    if not full_text:
        return []

    raw_ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END_RE.finditer(full_text):
        raw_ranges.append((cursor, match.end()))
        cursor = match.end()
    if cursor < len(full_text):
        raw_ranges.append((cursor, len(full_text)))

    sentences: list[TranscriptSentence] = []
    previous_end_ms = 0
    for raw_start, raw_end in raw_ranges:
        for chunk_start, chunk_end, is_fallback in _split_long_range(raw_start, raw_end, spans):
            sentence, _ = _sentence_from_range(
                transcript.video_id,
                full_text,
                spans,
                chunk_start,
                chunk_end,
                len(sentences),
                previous_end_ms,
                limit_duration=is_fallback,
            )
            if sentence is not None:
                sentences.append(sentence)
                previous_end_ms = sentence.end_ms

    return sentences


def _split_long_range(
    raw_start: int,
    raw_end: int,
    spans: list[_SourceSpan],
) -> list[tuple[int, int, bool]]:
    """Split long punctuation-free text at source-fragment boundaries.

    Short, normally punctuated ranges pass through unchanged. A fallback chunk
    ends before it would exceed 15 seconds or 320 characters whenever an
    original transcript-fragment boundary is available.
    """
    if raw_start >= raw_end:
        return []

    raw_duration_ms = _timestamp_for_char(spans, raw_end, is_end=True) - _timestamp_for_char(
        spans, raw_start, is_end=False
    )
    needs_fallback = raw_duration_ms > FALLBACK_MAX_DURATION_MS or raw_end - raw_start > FALLBACK_MAX_CHARACTERS

    boundaries = [
        span.char_end
        for span in spans
        if raw_start < span.char_end < raw_end
    ]
    if not boundaries:
        return [(raw_start, raw_end, needs_fallback)]

    chunks: list[tuple[int, int, bool]] = []
    chunk_start = raw_start
    previous_boundary = raw_start
    for boundary in boundaries:
        duration_ms = _timestamp_for_char(spans, boundary, is_end=True) - _timestamp_for_char(
            spans, chunk_start, is_end=False
        )
        character_count = boundary - chunk_start
        if duration_ms > FALLBACK_MAX_DURATION_MS or character_count > FALLBACK_MAX_CHARACTERS:
            chunk_end = previous_boundary if previous_boundary > chunk_start else boundary
            chunks.append((chunk_start, chunk_end, True))
            chunk_start = chunk_end
        previous_boundary = boundary

    if chunk_start < raw_end:
        chunks.append((chunk_start, raw_end, needs_fallback))
    return chunks


def sentence_at(sentences: list[TranscriptSentence], timestamp_ms: int) -> TranscriptSentence | None:
    """Return the sentence whose half-open interval contains ``timestamp_ms``."""
    for sentence in sentences:
        if sentence.start_ms <= timestamp_ms < sentence.end_ms:
            return sentence
    return None


def _join_segments(segments: list[TranscriptSegment]) -> tuple[str, list[_SourceSpan]]:
    parts: list[str] = []
    spans: list[_SourceSpan] = []
    offset = 0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if parts:
            offset += 1
        char_start = offset
        char_end = char_start + len(text)
        parts.append(text)
        spans.append(_SourceSpan(char_start, char_end, segment.start_ms, segment.end_ms))
        offset = char_end

    return " ".join(parts), spans


def _sentence_from_range(
    video_id: str,
    full_text: str,
    spans: list[_SourceSpan],
    raw_start: int,
    raw_end: int,
    sentence_number: int,
    previous_end_ms: int,
    limit_duration: bool,
) -> tuple[TranscriptSentence | None, int]:
    start = raw_start
    end = raw_end
    while start < end and full_text[start].isspace():
        start += 1
    while end > start and full_text[end - 1].isspace():
        end -= 1
    if start == end:
        return None, raw_end

    actual_start_ms = _timestamp_for_char(spans, start, is_end=False)
    start_ms = max(previous_end_ms, actual_start_ms)
    end_ms = max(start_ms + 1, _timestamp_for_char(spans, end, is_end=True))
    if limit_duration:
        end_ms = min(end_ms, start_ms + FALLBACK_MAX_DURATION_MS)
    return TranscriptSentence(
        sentence_id=f"{video_id}_S{sentence_number:06d}",
        start_ms=start_ms,
        end_ms=end_ms,
        text=full_text[start:end],
    ), raw_end


def _timestamp_for_char(spans: list[_SourceSpan], char_offset: int, *, is_end: bool) -> int:
    """Interpolate a character boundary to a source-fragment timestamp."""
    if not spans:
        return 0

    for span in spans:
        if span.char_start <= char_offset <= span.char_end:
            width = max(1, span.char_end - span.char_start)
            ratio = (char_offset - span.char_start) / width
            return round(span.start_ms + ratio * (span.end_ms - span.start_ms))

    if char_offset < spans[0].char_start:
        return spans[0].start_ms
    return spans[-1].end_ms if is_end else spans[-1].start_ms
