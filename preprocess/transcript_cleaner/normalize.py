"""Normalize legacy timestamped transcript TXT files to the cleaner JSONL contract."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


TIMESTAMP_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?\]\s*(.*?)\s*$"
)
VIDEO_ID_RE = re.compile(r"^(L\d{2}_V\d{3})")
FALLBACK_DURATION_MS = 5_000


def derive_video_id(filename: str) -> str:
    stem = Path(filename).stem
    match = VIDEO_ID_RE.match(stem)
    if match:
        return match.group(1)
    if re.fullmatch(r"[\w.-]+", stem):
        return stem.removesuffix("_Transcript")
    raise ValueError(f"cannot derive video_id from filename: {filename}")


def parse_timestamped_txt(path: Path) -> list[dict]:
    parsed: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = TIMESTAMP_RE.match(line.strip())
        if match is None:
            raise ValueError(f"{path}:{line_number}: invalid timestamped transcript line")
        hours, minutes, seconds = (int(match.group(index)) for index in (1, 2, 3))
        fraction = (match.group(4) or "").ljust(3, "0")
        milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000
        if fraction:
            milliseconds += int(fraction)
        text = match.group(5).strip()
        if not text:
            raise ValueError(f"{path}:{line_number}: transcript text is empty")
        if parsed and milliseconds < parsed[-1][0]:
            raise ValueError(f"{path}:{line_number}: timestamps are not ordered")
        parsed.append((milliseconds, text))

    if not parsed:
        raise ValueError(f"{path}: transcript contains no segments")
    return [
        {
            "start_time_ms": start,
            "end_time_ms": parsed[index + 1][0] if index + 1 < len(parsed) else start + FALLBACK_DURATION_MS,
            "text": text,
        }
        for index, (start, text) in enumerate(parsed)
    ]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def normalize_txt_directory(input_dir: Path, output_dir: Path, *, overwrite: bool = False) -> int:
    if not input_dir.is_dir():
        raise ValueError(f"transcript input directory does not exist: {input_dir}")
    converted = 0
    for source in sorted(input_dir.glob("*.txt")):
        target = output_dir / f"{derive_video_id(source.name)}.jsonl"
        if target.exists() and not overwrite:
            continue
        _write_jsonl_atomic(target, parse_timestamped_txt(source))
        converted += 1
    return converted
