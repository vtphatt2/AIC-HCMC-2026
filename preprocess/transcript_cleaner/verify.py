from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import TranscriptError


@dataclass(frozen=True, slots=True)
class VerificationReport:
    source: Path
    output: Path
    segment_count: int
    changed_text_count: int
    unchanged_text_count: int
    source_text_characters: int
    output_text_characters: int


def verify_output(source: Path, output: Path) -> VerificationReport:
    try:
        source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptError(f"Cannot parse source/output JSONL: {exc}") from exc

    if len(source_rows) != len(output_rows):
        raise TranscriptError(
            f"Row count changed: input={len(source_rows)}, output={len(output_rows)}"
        )

    changed = 0
    for index, (original, cleaned) in enumerate(
        zip(source_rows, output_rows, strict=True)
    ):
        if not isinstance(original, dict) or not isinstance(cleaned, dict):
            raise TranscriptError(f"Row {index} is not a JSON object")
        if set(original) != set(cleaned):
            raise TranscriptError(f"Row {index} field set changed")
        for key, value in original.items():
            if key != "text" and cleaned.get(key) != value:
                raise TranscriptError(f"Row {index} changed protected field {key}")
        text = cleaned.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptError(f"Row {index} has invalid output text")
        changed += original.get("text") != text

    if not output.read_bytes().endswith(b"\n"):
        raise TranscriptError("Output JSONL does not end with a newline")

    return VerificationReport(
        source=source,
        output=output,
        segment_count=len(source_rows),
        changed_text_count=changed,
        unchanged_text_count=len(source_rows) - changed,
        source_text_characters=sum(len(row["text"]) for row in source_rows),
        output_text_characters=sum(len(row["text"]) for row in output_rows),
    )
