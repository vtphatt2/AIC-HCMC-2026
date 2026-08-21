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

from app.data_provider import data_subdir

logger = logging.getLogger(__name__)

_LINE_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)")
_LAST_SEGMENT_FALLBACK_MS = 5000

# A handful of this dataset's segments are single-character caption
# fragments (~1% — see search_all_transcripts). Every fuzzy scorer,
# WRatio included, treats "is the shorter string fully contained in the
# longer one" as a strong signal — which a 1-2 char segment satisfies
# against nearly any query, scoring ~90 regardless of relevance. There's no
# scorer choice that fixes this (it's inherent to what "fuzzy" match means
# for a 1-char string); excluding segments too short to be a meaningful
# phrase match is the actual fix.
MIN_FUZZY_SEGMENT_LEN = 4


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

    @classmethod
    def from_jsonl(cls, path: Path, video_id: str | None = None) -> "Transcript":
        """Parses the pre-processed `{"start_time_ms", "end_time_ms", "text"}`
        JSONL format (see remote-server/app/services/transcript_jsonl_reader.py,
        which reads the same challenge_resources/data/transcripts output)."""
        vid = video_id or path.stem
        segments: list[TranscriptSegment] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            seg = json.loads(line)
            segments.append(TranscriptSegment(
                start_ms=seg["start_time_ms"], end_ms=seg["end_time_ms"], text=seg["text"],
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
        """Approximate matching, scored per segment via rapidfuzz's WRatio
        (0-100; tolerant of the query being a reworded or partial match
        rather than an exact phrase). WRatio, not partial_ratio: plain
        partial_ratio scores by how well the *shorter* string fits inside
        the longer one, so a one-word segment (some of this dataset's
        segments are caption fragments that short) trivially "fully
        matches" any query containing that word — WRatio blends in the
        overall length ratio so tiny segments stop winning by default.
        Segment granularity (each is already a few seconds of speech) is
        coarser than the char-offset windowing a fuzzy *substring* search
        would need, but good enough to locate content that doesn't match
        verbatim, without wiring up sliding windows over full_text."""
        query = query.strip()
        if not query or not self.segments:
            return []
        from rapidfuzz import fuzz

        lower_query = query.lower()
        matches = []
        for seg in self.segments:
            if len(seg.text.strip()) < MIN_FUZZY_SEGMENT_LEN:
                continue
            score = fuzz.WRatio(lower_query, seg.text.lower())
            if score >= threshold:
                matches.append(TranscriptMatch(
                    start_ms=seg.start_ms, end_ms=seg.end_ms, text=seg.text,
                    char_start=0, char_end=len(seg.text), score=score,
                ))
        return matches

    def _timestamp_for_char(self, char_offset: int) -> int:
        if not self.time_index:
            return 0
        offsets = [o for o, _ in self.time_index]
        i = bisect_right(offsets, char_offset) - 1
        i = max(0, min(i, len(self.time_index) - 1))
        return self.time_index[i][1]


def load_or_build_transcript(video_id: str) -> Transcript | None:
    """Loads the cached JSON if present and newer than the source file,
    otherwise parses the source and writes the cache. Prefers the
    pre-processed `{video_id}.jsonl` (current challenge_resources/data/transcripts
    format); falls back to the older `{video_id}_Transcript.txt` format.
    Returns None if neither exists for this video_id."""
    transcripts_dir = data_subdir("transcripts")
    jsonl_path = transcripts_dir / f"{video_id}.jsonl"
    txt_path = transcripts_dir / f"{video_id}_Transcript.txt"

    if jsonl_path.is_file():
        source_path, parse = jsonl_path, Transcript.from_jsonl
    elif txt_path.is_file():
        source_path, parse = txt_path, Transcript.from_txt
    else:
        return None

    cache_path = data_subdir("transcripts_processed") / f"{video_id}.json"
    if cache_path.is_file() and cache_path.stat().st_mtime >= source_path.stat().st_mtime:
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return Transcript.from_json(data)
        except Exception:
            logger.warning("Failed to load transcript cache for %s, rebuilding", video_id, exc_info=True)

    transcript = parse(source_path, video_id=video_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(transcript.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return transcript


# ── Search across every cached transcript ──────────────────────────────────
# The local (ZIP-mode) equivalent of remote-server's Postgres full-text
# search over 40-60s chunks — done here via Transcript.fuzzy_find over each
# video's own segments instead. No embedding model involved (topic
# classification and true semantic ranking both need one — out of scope
# locally), so results carry topic="" and are ranked by fuzzy score alone.
# Good enough to make the Transcripts search tab work locally at all, which
# it previously didn't (DataProvider.search_transcript_chunks returned []
# outside ENV_MODE=LOCAL).

_all_transcripts_cache: dict[str, Transcript] = {}
# (video_id, segment) pairs across every cached transcript, flattened once —
# ~154k segments over 814 videos. Scored in one batched rapidfuzz.process.extract
# call rather than Transcript.fuzzy_find's per-segment Python loop: at this
# count, per-call Python/function-call overhead dominates (a first cut of
# this feature measured ~30s/query looping fuzzy_find per video), where
# process.extract's batched C path comes back in well under a second.
_all_segments_cache: list[tuple[str, TranscriptSegment]] | None = None


def _all_video_ids_with_transcripts() -> list[str]:
    transcripts_dir = data_subdir("transcripts")
    if not transcripts_dir.is_dir():
        return []
    return sorted(p.stem for p in transcripts_dir.glob("*.jsonl"))


def _all_segments() -> list[tuple[str, TranscriptSegment]]:
    global _all_segments_cache
    if _all_segments_cache is None:
        pairs: list[tuple[str, TranscriptSegment]] = []
        for video_id in _all_video_ids_with_transcripts():
            transcript = _all_transcripts_cache.get(video_id)
            if transcript is None:
                transcript = load_or_build_transcript(video_id)
                if transcript is None:
                    continue
                _all_transcripts_cache[video_id] = transcript
            pairs.extend(
                (video_id, seg) for seg in transcript.segments
                if len(seg.text.strip()) >= MIN_FUZZY_SEGMENT_LEN
            )
        _all_segments_cache = pairs
    return _all_segments_cache


def search_all_transcripts(query: str, top_k: int = 100, threshold: float = 70.0) -> list[dict]:
    """Fuzzy-matches `query` against every cached video's transcript
    segments, ranks by score, and resolves each hit to its nearest indexed
    keyframe (for a thumbnail + frame_number) via numpy_vector_store.
    Shaped to match the frontend's TranscriptChunkResult exactly."""
    query = query.strip()
    if not query:
        return []
    from rapidfuzz import fuzz, process

    pairs = _all_segments()
    hits = process.extract(
        query, [seg.text for _, seg in pairs],
        scorer=fuzz.WRatio, limit=top_k, score_cutoff=threshold,
    )
    matches = [(pairs[idx][0], TranscriptMatch(
        start_ms=pairs[idx][1].start_ms, end_ms=pairs[idx][1].end_ms,
        text=pairs[idx][1].text, char_start=0, char_end=len(pairs[idx][1].text), score=score,
    )) for _text, score, idx in hits]

    from app.db import numpy_vector_store

    have_frames = numpy_vector_store.available()
    results = []
    for i, (video_id, m) in enumerate(matches):
        frame_number, timestamp_ms, youtube_id, frame_image_url = 0, None, "", ""
        if have_frames:
            mid_ms = (m.start_ms + m.end_ms) // 2
            # Keyframe sampling is scene-adaptive, not fixed-interval — gaps
            # of 30s+ are normal in a low-motion stretch (a lecture's static
            # talking-head shot, say), so a fixed time window around the
            # match can legitimately come back empty. Pull every keyframe
            # of the video instead (~a few hundred at most, see
            # per-video-frame-count check) and pick whichever is nearest —
            # frames_in_range's own `limit` keeps the *earliest* rows in a
            # range, not the *nearest* to a target, so limiting to a small
            # window would silently drop the very hit we're after.
            nearby = numpy_vector_store.frames_in_range(video_id, 0, 10**12, limit=2000)
            if nearby:
                nearest = min(nearby, key=lambda f: abs(f["timestamp_ms"] - mid_ms))
                frame_number = nearest["frame_number"]
                timestamp_ms = nearest["timestamp_ms"]
                youtube_id = nearest["youtube_id"] or ""
                frame_image_url = f"/api/zip-frame/{video_id}/{timestamp_ms}"
        results.append({
            "chunk_id": i,
            "video_id": video_id,
            "youtube_id": youtube_id,
            "topic": "",
            "start_time_ms": m.start_ms,
            "end_time_ms": m.end_ms,
            "text": m.text,
            "score": m.score / 100.0,
            "frame_image_url": frame_image_url,
            "frame_number": frame_number,
            "nearest_timestamp_ms": timestamp_ms,
        })
    return results
