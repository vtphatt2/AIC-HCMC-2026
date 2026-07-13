"""
Convert existing [HH:MM:SS] transcript .txt files to JSONL format.
Each line in the output is a JSON object:
    {"start_time_ms": 1500, "end_time_ms": 4500, "text": "..."}

Usage:
    python scripts/convert_transcripts_to_jsonl.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TIMESTAMP_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s+(.*)")
VIDEO_ID_RE = re.compile(r"^(L\d{2}_V\d{3})")
FALLBACK_DURATION_MS = 5000


def derive_video_id(filename: str) -> str | None:
    stem = Path(filename).stem
    match = VIDEO_ID_RE.match(stem)
    if match:
        return match.group(1)
    if re.match(r"^[\w-]+$", stem):
        return stem
    return None


def parse_transcript_file(filepath: Path) -> tuple[str, list[dict]]:
    video_id = derive_video_id(filepath.name)
    if not video_id:
        raise ValueError(f"Cannot derive video_id from filename: {filepath.name}")

    raw_lines = filepath.read_text(encoding="utf-8").splitlines()
    parsed = []
    for line in raw_lines:
        match = TIMESTAMP_RE.match(line.strip())
        if match:
            h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
            parsed.append((h, m, s, match.group(4).strip()))

    segments = []
    for i, (h, m, s, text) in enumerate(parsed):
        start_ms = (h * 3600 + m * 60 + s) * 1000
        if i + 1 < len(parsed):
            nh, nm, ns, _ = parsed[i + 1]
            end_ms = (nh * 3600 + nm * 60 + ns) * 1000
        else:
            end_ms = start_ms + FALLBACK_DURATION_MS

        segments.append({
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "text": text,
        })

    return video_id, segments


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Find transcript directories
    sample_transcripts = repo_root.parent / "AIC2026_sample" / "transcripts" / "transcripts"
    notebooks_transcripts = repo_root.parent / "notebooks" / "transcripts_sample"

    dirs = [sample_transcripts, notebooks_transcripts]
    converted = 0

    for transcripts_dir in dirs:
        if not transcripts_dir.is_dir():
            print(f"Skipping (not found): {transcripts_dir}")
            continue

        txt_files = sorted(transcripts_dir.glob("*_Transcript.txt"))
        if not txt_files:
            txt_files = sorted(transcripts_dir.glob("*.txt"))

        for filepath in txt_files:
            try:
                video_id, segments = parse_transcript_file(filepath)
            except ValueError as exc:
                print(f"SKIP {filepath.name}: {exc}")
                continue

            if not segments:
                print(f"SKIP {filepath.name}: no valid segments")
                continue

            jsonl_path = filepath.with_name(f"{video_id}.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")

            print(f"OK  {filepath.name} → {jsonl_path.name}  ({len(segments)} segments)")
            converted += 1

    print(f"\nDone. Converted {converted} files to JSONL.")


if __name__ == "__main__":
    main()
