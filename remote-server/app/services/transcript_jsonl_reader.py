import json
import os
from pathlib import Path

VIDEO_ID_RE = r"^(L\d{2}_V\d{3})"


def _find_transcripts_dir() -> Path | None:
    """Find the transcripts directory by scanning upward from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates = [
            parent / "AIC2026_sample" / "transcripts" / "transcripts",
            parent / "AIC2026_sample" / "transcripts",
            parent / "notebooks" / "transcripts_sample",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    return None


def _find_jsonl(video_id: str) -> Path | None:
    """Find a .jsonl file for the given video_id."""
    transcripts_dir = _find_transcripts_dir()
    if transcripts_dir is None:
        return None

    candidates = [
        transcripts_dir / f"{video_id}.jsonl",
        transcripts_dir / f"{video_id}_Transcript.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def read_transcript_jsonl(video_id: str) -> list[dict]:
    """Read transcript segments from a JSONL file.

    Each line is a JSON object: {"start_time_ms": ..., "end_time_ms": ..., "text": "..."}
    Returns a list of dicts with an added "id" field.
    """
    filepath = _find_jsonl(video_id)
    if filepath is None:
        return []

    segments: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                segment = json.loads(line)
            except json.JSONDecodeError:
                continue
            segment["id"] = f"{video_id}_{idx}"
            segment["video_id"] = video_id
            segments.append(segment)

    return segments


def read_transcript_jsonl_as_api(video_id: str) -> list[dict]:
    """Same as read_transcript_jsonl but returns only API-facing fields."""
    segments = read_transcript_jsonl(video_id)
    return [
        {
            "start_time_ms": s.get("start_time_ms", 0),
            "end_time_ms": s.get("end_time_ms", 0),
            "text": s.get("text", ""),
        }
        for s in segments
    ]
