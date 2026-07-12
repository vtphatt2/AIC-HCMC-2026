"""
Local transcript module — parses raw `[HH:MM:SS] fragment` transcript .txt
files (see notebooks/scrape_transcript.py) into a structured, queryable
form, with a JSON cache so parsing only happens once per video.

Scope note: this module intentionally does NOT implement transcript
*search* yet (semantic ranking, fuzzy-driven cross-video matching, fusion
with visual results) — that needs to be designed together with visual
search later. See the "FUTURE SEARCH INTERFACE" block at the bottom for
the documented, not-yet-built extension points. What IS built here are the
timing/lookup primitives needed to show transcript context for a moment we
already know about (e.g. a visual search result's video_id + timestamp_ms)
— and that future search work will also need.
"""
from __future__ import annotations

import json
import logging
import re
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path

from app.data_provider import sample_subdir

logger = logging.getLogger(__name__)

_LINE_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)")
_LAST_SEGMENT_FALLBACK_MS = 5000


@dataclass
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        return cls(start_ms=d["start_ms"], end_ms=d["end_ms"], text=d["text"], speaker=d.get("speaker"))


@dataclass
class TranscriptMatch:
    start_ms: int
    end_ms: int
    text: str
    char_start: int
    char_end: int
    score: float = 1.0


def _build_full_text_and_index(segments: list[TranscriptSegment]) -> tuple[str, list[tuple[int, int]]]:
    """full_text = every segment's text space-joined (so a lookup/search
    never gets cut off mid-line the way the raw wrapped .txt would).
    time_index = one (char_offset, timestamp_ms) breakpoint per segment
    boundary in full_text, letting any char offset resolve to a timestamp
    via interpolation — the primitive a future substring search needs to
    turn a text match into a (start_ms, end_ms)."""
    parts: list[str] = []
    index: list[tuple[int, int]] = []
    offset = 0
    for seg in segments:
        index.append((offset, seg.start_ms))
        parts.append(seg.text)
        offset += len(seg.text) + 1  # +1 for the space "".join(parts) will insert
    return " ".join(parts), index


class Transcript:
    """One video's transcript: raw segments plus the derived full_text and
    time_index that make timing lookups (and future substring search) cheap."""

    def __init__(self, video_id: str, segments: list[TranscriptSegment]):
        self.video_id = video_id
        self.segments = segments
        self.full_text, self.time_index = _build_full_text_and_index(segments)

    # ── Parsing ──────────────────────────────────────────────────────────
    @classmethod
    def from_txt(cls, path: Path, video_id: str | None = None) -> "Transcript":
        vid = video_id or path.stem.replace("_Transcript", "")
        lines = path.read_text(encoding="utf-8").splitlines()

        parsed: list[tuple[int, str, str | None]] = []
        for line in lines:
            m = _LINE_RE.match(line.strip())
            if not m:
                continue
            h, mnt, s, text = m.groups()
            start_ms = ((int(h) * 60 + int(mnt)) * 60 + int(s)) * 1000
            text = text.strip()
            speaker = None
            if text.startswith(">>"):
                speaker = "speaker"
                text = text[2:].strip()
            if text:
                parsed.append((start_ms, text, speaker))

        segments: list[TranscriptSegment] = []
        for i, (start_ms, text, speaker) in enumerate(parsed):
            end_ms = parsed[i + 1][0] if i + 1 < len(parsed) else start_ms + _LAST_SEGMENT_FALLBACK_MS
            segments.append(TranscriptSegment(
                start_ms=start_ms, end_ms=max(end_ms, start_ms + 1), text=text, speaker=speaker,
            ))

        return cls(vid, segments)

    # ── Serialization ────────────────────────────────────────────────────
    def to_json(self) -> dict:
        return {"video_id": self.video_id, "segments": [s.to_dict() for s in self.segments]}

    @classmethod
    def from_json(cls, data: dict) -> "Transcript":
        segments = [TranscriptSegment.from_dict(s) for s in data["segments"]]
        return cls(data["video_id"], segments)

    # ── Timing <-> text lookups ──────────────────────────────────────────
    def segment_at(self, timestamp_ms: int) -> TranscriptSegment | None:
        """Timing -> text: the segment covering timestamp_ms, or the
        nearest one if it falls in a gap between segments."""
        if not self.segments:
            return None
        for seg in self.segments:  # per-video lists are small (hundreds); linear scan is fine
            if seg.start_ms <= timestamp_ms < seg.end_ms:
                return seg
        return min(self.segments, key=lambda s: abs(s.start_ms - timestamp_ms))

    def text_window(self, timestamp_ms: int, radius_s: int = 15) -> str:
        """A few segments of context around a moment, for display."""
        radius_ms = radius_s * 1000
        window = [
            s for s in self.segments
            if s.end_ms >= timestamp_ms - radius_ms and s.start_ms <= timestamp_ms + radius_ms
        ]
        return " ".join(s.text for s in window)

    def find(self, substring: str) -> list[TranscriptMatch]:
        """Exact (case-insensitive) substring search over full_text, with
        timing resolved via time_index. Implemented now — trivial and
        useful (e.g. jump-to-exact-phrase) — unlike fuzzy/semantic matching."""
        if not substring:
            return []
        matches: list[TranscriptMatch] = []
        lower_text = self.full_text.lower()
        lower_sub = substring.lower()
        start = 0
        while True:
            idx = lower_text.find(lower_sub, start)
            if idx == -1:
                break
            char_end = idx + len(substring)
            matches.append(TranscriptMatch(
                start_ms=self._timestamp_for_char(idx),
                end_ms=self._timestamp_for_char(char_end),
                text=self.full_text[idx:char_end],
                char_start=idx,
                char_end=char_end,
                score=1.0,
            ))
            start = idx + 1
        return matches

    def fuzzy_find(self, query: str, threshold: float = 70.0) -> list[TranscriptMatch]:
        """NOT YET IMPLEMENTED — documented interface for future search
        work. Intended behavior: approximate substring matching over
        full_text (e.g. via rapidfuzz's partial_ratio sliding window), for
        locating content that doesn't match verbatim. Left as a stub since
        transcript *search* itself is deferred (see module docstring)."""
        raise NotImplementedError(
            "Transcript.fuzzy_find is a documented interface for future search work, not yet implemented."
        )

    def _timestamp_for_char(self, char_offset: int) -> int:
        if not self.time_index:
            return 0
        offsets = [o for o, _ in self.time_index]
        i = bisect_right(offsets, char_offset) - 1
        i = max(0, min(i, len(self.time_index) - 1))
        return self.time_index[i][1]


def load_or_build_transcript(video_id: str) -> Transcript | None:
    """Loads the cached JSON if present and newer than the source .txt,
    otherwise parses the .txt and writes the cache. Returns None if no
    transcript .txt exists for this video_id."""
    txt_path = sample_subdir("transcripts") / f"{video_id}_Transcript.txt"
    if not txt_path.is_file():
        return None

    cache_path = sample_subdir("transcripts_processed") / f"{video_id}.json"
    if cache_path.is_file() and cache_path.stat().st_mtime >= txt_path.stat().st_mtime:
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return Transcript.from_json(data)
        except Exception:
            logger.warning("Failed to load transcript cache for %s, rebuilding", video_id, exc_info=True)

    transcript = Transcript.from_txt(txt_path, video_id=video_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(transcript.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return transcript


# ── FUTURE SEARCH INTERFACE (not implemented) ──────────────────────────────
#
# This module deliberately stops at timing/lookup primitives. Real
# transcript *search* — needed to replace DataProvider.search_transcript_chunks's
# current `return []` in SAMPLE mode — would add, roughly:
#
#   @dataclass
#   class TranscriptChunk:
#       chunk_id: int
#       start_ms: int
#       end_ms: int
#       text: str
#       topic: str
#       embedding: list[float] | None
#
#   def build_chunks(transcript: Transcript, target_s=45, overlap=0.5) -> list[TranscriptChunk]:
#       """Sliding-window chunking over `transcript.segments`, mirroring
#       remote-server/scripts/index_transcripts.py's chunk_segments
#       (~40-60s windows, 50% overlap) so local results stay comparable to
#       production. Each chunk's `topic` = nearest of the same 14 fixed
#       Vietnamese labels via e5 cosine (port TOPICS + classify_topic from
#       remote-server/app/db/milvus_client.py)."""
#
#   def embed_chunks(chunks: list[TranscriptChunk]) -> None:
#       """Fills in each chunk's `embedding` using multilingual-e5-small
#       (already installed; same model remote-server/app/services/
#       transcript_search.py uses), "passage: " prefixed per the e5
#       convention. Query-time embedding uses the "query: " prefix."""
#
#   def search_all_transcripts(query: str, top_k: int, topic_filter: str | None = None) -> list[dict]:
#       """Embeds `query`, ranks all cached videos' chunks by cosine
#       similarity (brute-force numpy — dataset is small enough that no
#       vector DB is needed locally), shapes hits into the existing
#       TranscriptChunkResult dict shape, and is meant to be dropped
#       straight into DataProvider.search_transcript_chunks's SAMPLE-mode
#       branch. transcript_fusion_strategy.py already prefers real
#       transcript_chunks over its token-overlap fallback whenever they're
#       present, so no strategy changes would be needed when this lands."""
#
# Transcript.fuzzy_find() above is the one piece of this that already has
# a defined signature, since rapidfuzz-based approximate matching is a
# small, self-contained addition whenever this gets picked up.
