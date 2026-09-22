"""Transcript-cleaning stage and generic concurrent process orchestration."""
from __future__ import annotations

import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


def lot_video_id_regex(data_id: str) -> str | None:
    """Map old lot IDs or a new numeric range to its exact metadata IDs."""

    range_match = re.fullmatch(r"([A-Za-z][A-Za-z_-]*)(\d+)-([A-Za-z][A-Za-z_-]*)(\d+)", data_id)
    if range_match is not None:
        start_prefix, start_number, end_prefix, end_number = range_match.groups()
        start, end = int(start_number), int(end_number)
        if start_prefix.casefold() == end_prefix.casefold() and start <= end and end - start < 10_000:
            width = max(len(start_number), len(end_number))
            identifiers = "|".join(
                re.escape(f"{start_prefix}{number:0{width}d}")
                for number in range(start, end + 1)
            )
            return rf"^(?:{identifiers})$"

    match = re.fullmatch(r"(L\d+)_([a-z])", data_id, flags=re.IGNORECASE)
    if match is None:
        lot = re.match(r"^(L\d+)", data_id, flags=re.IGNORECASE)
        return rf"^{re.escape(lot.group(1))}_" if lot else None
    block = ord(match.group(2).lower()) - ord("a")
    return rf"^{re.escape(match.group(1))}_V{block}\d{{2}}$"


@dataclass(frozen=True)
class TranscriptStageConfig:
    """Arguments for the vendored, resumable Gemini transcript cleaner."""

    input_path: Path
    output_dir: Path = Path("clean_transcript")
    state_dir: Path = Path(".clean_transcript_state")
    model: str = "gemini-3.5-flash-lite"
    payload_format: str = "json-segments"
    api_key_env: str = "GEMINI_API_KEY"
    timeout: float = 120.0
    max_attempts: int = 3
    max_output_tokens: int = 65_536
    max_input_tokens: int = 100_000
    concurrency: int = 8
    large_request_threshold: int = 10_000
    large_request_concurrency: int = 2
    pattern: str = "*.jsonl"
    limit: int = 0
    rpm: int = 0
    tpm: int = 0
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("transcript concurrency must be positive")
        if self.large_request_concurrency < 1:
            raise ValueError("large request concurrency must be positive")
        if self.limit < 0 or self.rpm < 0 or self.tpm < 0:
            raise ValueError("limit, rpm, and tpm cannot be negative")
        if self.payload_format not in {"json-segments", "marker-text"}:
            raise ValueError("unsupported transcript payload format")


def build_transcript_command(
    config: TranscriptStageConfig,
    *,
    python_executable: str | None = None,
) -> list[str]:
    command = [
        python_executable or sys.executable,
        "-m",
        "preprocess.transcript_cleaner.cli",
        str(config.input_path),
        "--output-dir", str(config.output_dir),
        "--state-dir", str(config.state_dir),
        "--model", config.model,
        "--payload-format", config.payload_format,
        "--api-key-env", config.api_key_env,
        "--timeout", str(config.timeout),
        "--max-attempts", str(config.max_attempts),
        "--max-output-tokens", str(config.max_output_tokens),
        "--max-input-tokens", str(config.max_input_tokens),
        "--concurrency", str(config.concurrency),
        "--large-request-threshold", str(config.large_request_threshold),
        "--large-request-concurrency", str(config.large_request_concurrency),
        "--pattern", config.pattern,
        "--limit", str(config.limit),
        "--rpm", str(config.rpm),
        "--tpm", str(config.tpm),
    ]
    if config.overwrite:
        command.append("--overwrite")
    return command


@dataclass(frozen=True)
class CollectionCleaningConfig:
    """One metadata-to-clean-transcript workflow executed in its own process."""

    metadata_path: Path
    raw_dir: Path
    output_dir: Path
    state_dir: Path
    collection_concurrency: int = 8
    cleaning_concurrency: int = 8
    model: str = "gemini-3.5-flash-lite"
    rpm: int = 0
    tpm: int = 0
    video_id_regex: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.collection_concurrency < 1 or self.cleaning_concurrency < 1:
            raise ValueError("collection and cleaning concurrency must be positive")


def build_collection_cleaning_command(
    config: CollectionCleaningConfig,
    *,
    python_executable: str | None = None,
) -> list[str]:
    command = [
        python_executable or sys.executable,
        "-m", "preprocess.transcript_cleaner.workflow",
        "collect-clean", str(config.metadata_path),
        "--raw-dir", str(config.raw_dir),
        "--output-dir", str(config.output_dir),
        "--state-dir", str(config.state_dir),
        "--collection-concurrency", str(config.collection_concurrency),
        "--cleaning-concurrency", str(config.cleaning_concurrency),
        "--model", config.model,
        "--rpm", str(config.rpm),
        "--tpm", str(config.tpm),
    ]
    if config.video_id_regex is not None:
        command.extend(("--video-id-regex", config.video_id_regex))
    if config.overwrite:
        command.append("--overwrite")
    return command


def run_commands_concurrently(
    commands: Mapping[str, Sequence[str]],
    *,
    popen: Callable[..., object] = subprocess.Popen,
) -> dict[str, int]:
    """Start every independent stage before waiting for any one of them."""

    processes: dict[str, object] = {}
    try:
        for name, command in commands.items():
            processes[name] = popen(list(command))
        return {name: int(process.wait()) for name, process in processes.items()}
    except BaseException:
        for process in processes.values():
            if getattr(process, "returncode", None) is None:
                process.terminate()
        for process in processes.values():
            if getattr(process, "returncode", None) is None:
                try:
                    process.wait()
                except BaseException:
                    pass
        raise
