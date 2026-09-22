"""Collect YouTube captions from media-info JSON without extracting metadata ZIPs."""
from __future__ import annotations

import json
import os
import random
import re
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlparse


YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
NON_RETRYABLE_ERROR_NAMES = {
    "InvalidVideoId", "NoTranscriptFound", "TranscriptsDisabled", "VideoUnavailable",
}


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    metadata_path: Path
    output_dir: Path
    languages: tuple[str, ...] = ("vi", "en")
    concurrency: int = 8
    min_interval_seconds: float = 0.5
    max_attempts: int = 5
    backoff_base_seconds: float = 2.0
    video_id_regex: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("collection concurrency must be positive")
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not self.languages:
            raise ValueError("at least one transcript language is required")
        if self.video_id_regex is not None:
            re.compile(self.video_id_regex)


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    video_id: str
    youtube_id: str | None


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    discovered: int
    completed: int
    skipped: int
    failed: int
    failure_log: str | None
    missing_ids_log: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_youtube_id(url: str) -> str | None:
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.")
    candidate: str | None = None
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif host in {"youtube.com", "m.youtube.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        elif len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            candidate = parts[1]
    return candidate if candidate and YOUTUBE_ID_RE.fullmatch(candidate) else None


def _metadata_rows(path: Path) -> Iterable[tuple[str, bytes]]:
    if path.is_dir():
        for item in sorted(path.rglob("*.json")):
            yield item.stem, item.read_bytes()
        return
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".json") and not name.endswith("/"):
                    yield PurePosixPath(name).stem, archive.read(name)
        return
    if path.is_file() and path.suffix.lower() == ".json":
        yield path.stem, path.read_bytes()
        return
    raise ValueError(f"metadata source must be a JSON file, directory, or ZIP: {path}")


def discover_metadata(path: Path, video_id_regex: str | None = None) -> list[MetadataRecord]:
    records: list[MetadataRecord] = []
    seen: set[str] = set()
    selected = re.compile(video_id_regex) if video_id_regex else None
    for video_id, payload in _metadata_rows(path):
        if selected is not None and selected.search(video_id) is None:
            continue
        if video_id in seen:
            raise ValueError(f"duplicate metadata video_id: {video_id}")
        seen.add(video_id)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError(f"metadata for {video_id} is not a JSON object")
        url = next(
            (data.get(key) for key in ("watch_url", "video_link", "url") if isinstance(data.get(key), str)),
            "",
        )
        youtube_id = extract_youtube_id(url)
        records.append(MetadataRecord(video_id, youtube_id))
    return records


class _RequestPacer:
    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def _build_default_fetcher() -> Callable[[str, tuple[str, ...]], Sequence]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError(
            "youtube-transcript-api is required for collection; install preprocess/requirements.txt"
        ) from exc
    def fetch(youtube_id: str, languages: tuple[str, ...]):
        return YouTubeTranscriptApi().fetch(youtube_id, languages=list(languages)).snippets

    return fetch


def _snippet_value(snippet, field: str):
    if isinstance(snippet, Mapping):
        return snippet[field]
    return getattr(snippet, field)


def _rows_from_snippets(snippets: Sequence) -> list[dict]:
    rows: list[dict] = []
    for index, snippet in enumerate(snippets):
        start_ms = round(float(_snippet_value(snippet, "start")) * 1000)
        if index + 1 < len(snippets):
            end_ms = round(float(_snippet_value(snippets[index + 1], "start")) * 1000)
        else:
            end_ms = start_ms + 5_000
        text = str(_snippet_value(snippet, "text")).strip()
        if text:
            rows.append({"start_time_ms": start_ms, "end_time_ms": end_ms, "text": text})
    if not rows:
        raise ValueError("caption response contains no non-empty segments")
    return rows


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


def _write_missing_ids_atomic(path: Path, video_ids: Iterable[str]) -> None:
    """Publish a one-ID-per-line retry list without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for video_id in sorted(set(video_ids)):
                handle.write(f"{video_id}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def collect_transcripts(
    config: CollectionConfig,
    *,
    fetcher: Callable[[str, tuple[str, ...]], Sequence] | None = None,
) -> CollectionSummary:
    records = discover_metadata(config.metadata_path, config.video_id_regex)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _build_default_fetcher()
    pacer = _RequestPacer(config.min_interval_seconds)
    failures: list[dict] = []
    skipped = 0
    pending: list[MetadataRecord] = []
    for record in records:
        target = config.output_dir / f"{record.video_id}.jsonl"
        if target.exists() and not config.overwrite:
            skipped += 1
        elif record.youtube_id is None:
            failures.append({
                "video_id": record.video_id,
                "youtube_id": None,
                "error": "metadata has no supported YouTube URL",
            })
        else:
            pending.append(record)

    def collect_one(record: MetadataRecord) -> tuple[MetadataRecord, list[dict]]:
        assert record.youtube_id is not None
        last_error: Exception | None = None
        for attempt in range(1, config.max_attempts + 1):
            pacer.wait()
            try:
                return record, _rows_from_snippets(fetch(record.youtube_id, config.languages))
            except Exception as exc:
                last_error = exc
                if type(exc).__name__ in NON_RETRYABLE_ERROR_NAMES or attempt == config.max_attempts:
                    break
                ceiling = config.backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(random.uniform(0.0, ceiling))
        assert last_error is not None
        raise last_error

    completed = 0
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        future_to_record = {executor.submit(collect_one, record): record for record in pending}
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                _, rows = future.result()
                _write_jsonl_atomic(config.output_dir / f"{record.video_id}.jsonl", rows)
                completed += 1
            except Exception as exc:
                failures.append({
                    "video_id": record.video_id,
                    "youtube_id": record.youtube_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    # Keep diagnostics outside the cleaner's ``*.jsonl`` input glob.
    failure_path = config.output_dir / "collection_failures.ndjson"
    if failures:
        _write_jsonl_atomic(failure_path, failures)
    else:
        failure_path.unlink(missing_ok=True)
    missing_ids_path = config.output_dir / "missing_transcript_ids.txt"
    if failures:
        _write_missing_ids_atomic(missing_ids_path, (str(item["video_id"]) for item in failures))
    else:
        missing_ids_path.unlink(missing_ok=True)
    return CollectionSummary(
        discovered=len(records), completed=completed, skipped=skipped,
        failed=len(failures), failure_log=str(failure_path) if failures else None,
        missing_ids_log=str(missing_ids_path) if failures else None,
    )
