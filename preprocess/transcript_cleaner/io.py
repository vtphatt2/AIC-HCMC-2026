from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .models import Segment, TranscriptError


REQUIRED_FIELDS = {"start_time_ms", "end_time_ms", "text"}


def read_jsonl(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    previous_start: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TranscriptError(f"{path}:{line_number}: empty JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TranscriptError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TranscriptError(f"{path}:{line_number}: row must be an object")
            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                raise TranscriptError(
                    f"{path}:{line_number}: missing fields {sorted(missing)}"
                )

            start = row["start_time_ms"]
            end = row["end_time_ms"]
            text = row["text"]
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise TranscriptError(
                    f"{path}:{line_number}: start_time_ms must be a non-negative integer"
                )
            if isinstance(end, bool) or not isinstance(end, int) or end < start:
                raise TranscriptError(
                    f"{path}:{line_number}: end_time_ms must be an integer >= start_time_ms"
                )
            if not isinstance(text, str) or not text.strip():
                raise TranscriptError(f"{path}:{line_number}: text must be non-empty")
            if previous_start is not None and start < previous_start:
                raise TranscriptError(f"{path}:{line_number}: timestamps are not ordered")

            segments.append(
                Segment(
                    index=len(segments),
                    start_time_ms=start,
                    end_time_ms=end,
                    text=text,
                    raw=row,
                )
            )
            previous_start = start

    if not segments:
        raise TranscriptError(f"{path}: transcript is empty")
    return segments


def discover_inputs(input_dir: Path, pattern: str = "*.jsonl") -> list[Path]:
    if not input_dir.is_dir():
        raise TranscriptError(f"Input directory does not exist: {input_dir}")
    # Path.glob("*.jsonl") deliberately excludes the Windows :Zone.Identifier sidecars.
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def validate_existing_output(path: Path, source: list[Segment]) -> bool:
    try:
        output = read_jsonl(path)
    except (OSError, TranscriptError):
        return False
    if len(output) != len(source):
        return False
    for original, cleaned in zip(source, output, strict=True):
        if original.start_time_ms != cleaned.start_time_ms:
            return False
        if original.end_time_ms != cleaned.end_time_ms:
            return False
        if set(original.raw) != set(cleaned.raw):
            return False
        for key, value in original.raw.items():
            if key != "text" and cleaned.raw.get(key) != value:
                return False
    return True


def write_clean_jsonl_atomic(
    path: Path, source: Iterable[Segment], texts_by_index: dict[int, str]
) -> None:
    source_list = list(source)
    expected = set(range(len(source_list)))
    if set(texts_by_index) != expected:
        missing = sorted(expected.difference(texts_by_index))[:10]
        extra = sorted(set(texts_by_index).difference(expected))[:10]
        raise TranscriptError(f"Cannot merge output; missing={missing}, extra={extra}")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for segment in source_list:
                cleaned_text = texts_by_index[segment.index]
                if not isinstance(cleaned_text, str) or not cleaned_text.strip():
                    raise TranscriptError(
                        f"Segment {segment.index} has invalid cleaned text"
                    )
                row = dict(segment.raw)
                row["text"] = cleaned_text.strip()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
