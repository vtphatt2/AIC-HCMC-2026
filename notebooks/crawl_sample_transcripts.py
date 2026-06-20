"""Crawl YouTube transcripts for the 5 AIC2026 sample videos."""
import csv
import os
import re
import sys
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# ── Config ────────────────────────────────────────────────────────────────────
METADATA_DIR = Path(__file__).resolve().parent.parent / "AIC2026_sample" / "metadata"
# Handle nested layout: metadata/metadata/
if not any(METADATA_DIR.glob("*.json")):
    METADATA_DIR = METADATA_DIR / "metadata"
OUTPUT_DIR = Path(__file__).resolve().parent / "transcripts_sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LANGUAGES = ["vi", "en"]


def extract_yt_id(url: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def main() -> None:
    # Collect video IDs and YouTube links from metadata JSON files
    videos = []
    for meta_path in sorted(METADATA_DIR.glob("*.json")):
        import json

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        video_id = meta_path.stem
        yt_link = data.get("video_link", "")
        videos.append((video_id, yt_link))

    print(f"Found {len(videos)} videos from metadata.\n")

    success = 0
    for i, (video_id, yt_link) in enumerate(videos, 1):
        yt_id = extract_yt_id(yt_link)
        if not yt_id:
            print(f"  [{i}/{len(videos)}] FAIL {video_id} — invalid URL: {yt_link[:60]}")
            continue

        output_path = OUTPUT_DIR / f"{video_id}_Transcript.txt"
        if output_path.exists():
            print(f"  [{i}/{len(videos)}] SKIP {video_id} — already exists")
            success += 1
            continue

        try:
            result = YouTubeTranscriptApi().fetch(yt_id, languages=LANGUAGES)
            lines = [
                f"{format_timestamp(s.start)} {s.text}"
                for s in result.snippets
            ]
            output_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  [{i}/{len(videos)}] OK   {video_id} — {len(result.snippets)} segments")
            success += 1
        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"  [{i}/{len(videos)}] FAIL {video_id} — no transcript available")
        except VideoUnavailable:
            print(f"  [{i}/{len(videos)}] FAIL {video_id} — video unavailable")
        except Exception as exc:
            print(f"  [{i}/{len(videos)}] FAIL {video_id} — {type(exc).__name__}: {exc}")

    print(f"\nSuccess: {success}/{len(videos)}")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
