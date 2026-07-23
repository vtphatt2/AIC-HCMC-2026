import json
from pathlib import Path

from scripts.convert_transcripts_to_jsonl import parse_transcript_file
from scripts.sample_paths import default_sample_root, sample_subdir


def _transcripts_dir() -> Path:
    remote_root = Path(__file__).resolve().parents[2]
    return sample_subdir(default_sample_root(remote_root.parent), "transcripts")


def _find_jsonl(video_id: str) -> Path | None:
    """Find a .jsonl file for the given video_id."""
    transcripts_dir = _transcripts_dir()
    candidates = [
        transcripts_dir / f"{video_id}.jsonl",
        transcripts_dir / f"{video_id}_Transcript.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _find_txt(video_id: str) -> Path | None:
    transcripts_dir = _transcripts_dir()
    for path in (
        transcripts_dir / f"{video_id}_Transcript.txt",
        transcripts_dir / f"{video_id}.txt",
    ):
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
        txt_path = _find_txt(video_id)
        if txt_path is None:
            return []
        _, segments = parse_transcript_file(txt_path)
        return [
            {**segment, "id": f"{video_id}_{index}", "video_id": video_id}
            for index, segment in enumerate(segments)
        ]

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


def transcript_response(video_id: str) -> dict | None:
    """Build the transcript payload shared by the HTTP route and unit tests."""
    segments = read_transcript_jsonl_as_api(video_id)
    if not segments:
        return None
    return {
        "video_id": video_id,
        "segments": [
            {
                "start_ms": segment["start_time_ms"],
                "end_ms": segment["end_time_ms"],
                "text": segment["text"],
                "speaker": None,
            }
            for segment in segments
        ],
    }
