from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import read_jsonl, validate_existing_output, write_clean_jsonl_atomic
from .models import Chunk, CleanedChunk, FileJob, TranscriptError, WorkItem
from .state import StateStore, make_job_fingerprint, sha256_file
from .video_request import VideoRequestConfig, create_video_request


NON_SPEECH_MARKER = re.compile(r"^\s*\[[^\[\]\n]+\]\s*$")


def preserve_by_code(text: str) -> bool:
    stripped = text.strip()
    return bool(NON_SPEECH_MARKER.fullmatch(stripped)) or len(stripped) <= 1


class Cleaner(Protocol):
    async def clean(self, chunk: Chunk) -> CleanedChunk: ...


@dataclass(frozen=True, slots=True)
class RunSummary:
    discovered_files: int
    completed_files: int
    skipped_files: int
    failed_files: int
    api_chunks: int
    started_api_work_items: int
    resumed_chunks: int
    stopped_early: bool
    stop_reason: str | None


class TranscriptPipeline:
    def __init__(
        self,
        *,
        cleaner: Cleaner,
        state_store: StateStore,
        output_dir: Path,
        request_config: VideoRequestConfig,
        model: str,
        concurrency: int,
        overwrite: bool = False,
        generation_settings: dict[str, Any] | None = None,
        include_non_speech_markers: bool = False,
        large_request_threshold: int = 10_000,
        large_request_concurrency: int = 2,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self._cleaner = cleaner
        self._state = state_store
        self._output_dir = output_dir
        self._request_config = request_config
        self._model = model
        self._concurrency = concurrency
        self._overwrite = overwrite
        self._generation_settings = generation_settings or {}
        self._include_non_speech_markers = include_non_speech_markers
        self._large_request_threshold = large_request_threshold
        self._large_request_semaphore = asyncio.Semaphore(
            max(1, min(concurrency, large_request_concurrency))
        )
        self._fatal_error: Exception | None = None
        self._started_api_work_items = 0

    async def run(self, sources: list[Path]) -> RunSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[FileJob] = []
        skipped = 0
        resumed_chunks = 0

        for source_path in sources:
            segments = read_jsonl(source_path)
            output_path = self._output_dir / source_path.name
            source_hash = sha256_file(source_path)
            fingerprint = make_job_fingerprint(
                source_hash=source_hash,
                model=self._model,
                request_config=self._request_config,
                generation_settings=self._generation_settings,
            )
            if output_path.exists() and not self._overwrite:
                if validate_existing_output(
                    output_path, segments
                ) and self._state.is_completed(
                    source_name=source_path.name,
                    job_fingerprint=fingerprint,
                    output_path=output_path,
                ):
                    skipped += 1
                    continue
                raise TranscriptError(
                    f"Refusing to overwrite invalid/incomplete output {output_path}; "
                    "inspect it or pass --overwrite"
                )

            if self._include_non_speech_markers:
                cleanable = list(segments)
                preserved: dict[int, str] = {}
            else:
                cleanable = [
                    segment
                    for segment in segments
                    if not preserve_by_code(segment.text)
                ]
                preserved = {
                    segment.index: segment.text
                    for segment in segments
                    if preserve_by_code(segment.text)
                }
            # A Chunk is the internal persisted work-unit type. In this strategy it
            # always represents the entire video's cleanable text and has id 0.
            chunks = (
                [create_video_request(cleanable, self._request_config)]
                if cleanable
                else []
            )
            self._state.initialize_job(
                source_name=source_path.name,
                job_fingerprint=fingerprint,
                source_hash=source_hash,
                model=self._model,
                request_config=self._request_config,
                chunk_count=len(chunks),
                generation_settings=self._generation_settings,
            )
            job = FileJob(
                source_path=source_path,
                output_path=output_path,
                segments=segments,
                chunks=chunks,
                job_fingerprint=fingerprint,
                preserved_texts=preserved,
            )
            for chunk in chunks:
                cached = self._state.load_chunk(
                    source_name=source_path.name,
                    job_fingerprint=fingerprint,
                    chunk=chunk,
                )
                if cached is not None:
                    job.results[chunk.chunk_id] = cached
                    resumed_chunks += 1
            jobs.append(job)

        queue: asyncio.Queue[WorkItem | None] = asyncio.Queue()
        api_chunks = 0
        pending = [
            WorkItem(job=job, chunk=chunk)
            for job in jobs
            for chunk in job.chunks
            if chunk.chunk_id not in job.results
        ]
        # Shortest-job-first maximizes completed checkpoints before daily quota and
        # avoids starting the batch with several exceptionally large generations.
        pending.sort(key=lambda item: item.chunk.estimated_input_tokens)
        for item in pending:
            await queue.put(item)
            api_chunks += 1

        workers = [
            asyncio.create_task(self._worker(queue), name=f"gemini-worker-{index}")
            for index in range(min(self._concurrency, max(1, api_chunks)))
        ]
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        completed = 0
        failed = 0
        for job in jobs:
            if job.errors:
                failed += 1
                continue
            try:
                merged = dict(job.preserved_texts)
                for chunk in sorted(job.chunks, key=lambda value: value.chunk_id):
                    result = job.results.get(chunk.chunk_id)
                    if result is None:
                        raise TranscriptError(
                            f"Missing result for {job.source_path.name} chunk {chunk.chunk_id}"
                        )
                    overlap = set(merged).intersection(result.texts_by_index)
                    if overlap:
                        raise TranscriptError(
                            f"Duplicate segment ids during merge: {sorted(overlap)[:10]}"
                        )
                    merged.update(result.texts_by_index)
                write_clean_jsonl_atomic(job.output_path, job.segments, merged)
                if not validate_existing_output(job.output_path, job.segments):
                    raise TranscriptError(
                        f"Post-write validation failed for {job.output_path}"
                    )
                self._state.mark_completed(
                    source_name=job.source_path.name,
                    job_fingerprint=job.job_fingerprint,
                    output_path=job.output_path,
                )
                completed += 1
                self._append_event_safely(
                    {"event": "file_completed", "file": job.source_path.name}
                )
            except Exception as exc:
                failed += 1
                self._append_event_safely(
                    {
                        "event": "file_failed",
                        "file": job.source_path.name,
                        "error": str(exc),
                    }
                )

        return RunSummary(
            discovered_files=len(sources),
            completed_files=completed,
            skipped_files=skipped,
            failed_files=failed,
            api_chunks=api_chunks,
            started_api_work_items=self._started_api_work_items,
            resumed_chunks=resumed_chunks,
            stopped_early=self._fatal_error is not None,
            stop_reason=str(self._fatal_error) if self._fatal_error else None,
        )

    async def _worker(self, queue: asyncio.Queue[WorkItem | None]) -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                try:
                    if self._fatal_error is not None:
                        raise RuntimeError(
                            f"Skipped because a fatal API error stopped the run: "
                            f"{self._fatal_error}"
                        )
                    self._started_api_work_items += 1
                    if (
                        item.chunk.estimated_input_tokens
                        >= self._large_request_threshold
                    ):
                        async with self._large_request_semaphore:
                            result = await self._cleaner.clean(item.chunk)
                    else:
                        result = await self._cleaner.clean(item.chunk)
                    self._state.save_chunk(
                        source_name=item.job.source_path.name,
                        job_fingerprint=item.job.job_fingerprint,
                        chunk=item.chunk,
                        result=result,
                    )
                    item.job.results[item.chunk.chunk_id] = result
                    self._append_event_safely(
                        {
                            "event": "chunk_completed",
                            "file": item.job.source_path.name,
                            "chunk_id": item.chunk.chunk_id,
                            "attempts": result.attempts,
                            "latency_seconds": result.latency_seconds,
                            "estimated_input_tokens": item.chunk.estimated_input_tokens,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "fallback_segment_count": result.fallback_segment_count,
                        }
                    )
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    if (
                        status_code in {400, 401, 403, 404, 429}
                        and self._fatal_error is None
                    ):
                        self._fatal_error = exc
                        abort = getattr(self._cleaner, "abort", None)
                        if callable(abort):
                            abort()
                    item.job.errors[item.chunk.chunk_id] = exc
                    self._append_event_safely(
                        {
                            "event": "chunk_failed",
                            "file": item.job.source_path.name,
                            "chunk_id": item.chunk.chunk_id,
                            "error": str(exc),
                        }
                    )
            finally:
                queue.task_done()

    def _append_event_safely(self, event: dict[str, Any]) -> None:
        try:
            self._state.append_event(event)
        except OSError:
            # Checkpoint/output writes remain authoritative. Metrics must never kill
            # a worker and leave queue.join() waiting forever.
            pass
