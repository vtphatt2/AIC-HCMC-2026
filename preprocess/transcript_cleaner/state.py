from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fcntl

from .models import Chunk, CleanedChunk
from .prompt import PROMPT_VERSION
from .video_request import VideoRequestConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_job_fingerprint(
    *,
    source_hash: str,
    model: str,
    request_config: VideoRequestConfig,
    generation_settings: dict[str, Any] | None = None,
) -> str:
    payload = {
        "source_hash": source_hash,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "request_config": asdict(request_config),
        "generation_settings": generation_settings or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _job_dir(self, source_name: str, job_fingerprint: str) -> Path:
        return self.root / source_name / job_fingerprint

    @contextmanager
    def exclusive_run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".run.lock").open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "Another transcript-cleaner process is already using this state directory"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize_job(
        self,
        *,
        source_name: str,
        job_fingerprint: str,
        source_hash: str,
        model: str,
        request_config: VideoRequestConfig,
        chunk_count: int,
        generation_settings: dict[str, Any] | None = None,
    ) -> None:
        directory = self._job_dir(source_name, job_fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "source_name": source_name,
            "source_hash": source_hash,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "request_config": asdict(request_config),
            "chunk_count": chunk_count,
            "generation_settings": generation_settings or {},
        }
        path = directory / "job.json"
        if path.exists():
            try:
                if json.loads(path.read_text(encoding="utf-8")) == manifest:
                    return
            except (OSError, json.JSONDecodeError):
                pass
        _write_json_atomic(path, manifest)

    def load_chunk(
        self, *, source_name: str, job_fingerprint: str, chunk: Chunk
    ) -> CleanedChunk | None:
        path = (
            self._job_dir(source_name, job_fingerprint)
            / "chunks"
            / f"{chunk.chunk_id:06d}.json"
        )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("chunk_fingerprint") != chunk.fingerprint:
                return None
            raw_items = data["items"]
            if not isinstance(raw_items, list):
                return None
            texts: dict[int, str] = {}
            for item in raw_items:
                item_id = item["id"]
                text = item["text"]
                if (
                    isinstance(item_id, bool)
                    or not isinstance(item_id, int)
                    or not isinstance(text, str)
                    or not text.strip()
                    or item_id in texts
                ):
                    return None
                texts[item_id] = text
            expected = {segment.index for segment in chunk.segments}
            if set(texts) != expected:
                return None
            return CleanedChunk(
                chunk_id=chunk.chunk_id,
                texts_by_index=texts,
                attempts=int(data.get("attempts", 0)),
                input_tokens=_optional_int(data.get("input_tokens")),
                output_tokens=_optional_int(data.get("output_tokens")),
                latency_seconds=_optional_float(data.get("latency_seconds")),
                fallback_segment_count=int(data.get("fallback_segment_count", 0)),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save_chunk(
        self,
        *,
        source_name: str,
        job_fingerprint: str,
        chunk: Chunk,
        result: CleanedChunk,
    ) -> None:
        path = (
            self._job_dir(source_name, job_fingerprint)
            / "chunks"
            / f"{chunk.chunk_id:06d}.json"
        )
        data = {
            "chunk_id": chunk.chunk_id,
            "chunk_fingerprint": chunk.fingerprint,
            "attempts": result.attempts,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": result.latency_seconds,
            "fallback_segment_count": result.fallback_segment_count,
            "items": [
                {"id": item_id, "text": result.texts_by_index[item_id]}
                for item_id in sorted(result.texts_by_index)
            ],
        }
        _write_json_atomic(path, data)

    def append_event(self, event: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def mark_completed(
        self,
        *,
        source_name: str,
        job_fingerprint: str,
        output_path: Path,
    ) -> None:
        _write_json_atomic(
            self._job_dir(source_name, job_fingerprint) / "completed.json",
            {"output_hash": sha256_file(output_path)},
        )

    def is_completed(
        self,
        *,
        source_name: str,
        job_fingerprint: str,
        output_path: Path,
    ) -> bool:
        try:
            marker = json.loads(
                (
                    self._job_dir(source_name, job_fingerprint) / "completed.json"
                ).read_text(encoding="utf-8")
            )
            return marker.get("output_hash") == sha256_file(output_path)
        except (OSError, json.JSONDecodeError):
            return False


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
