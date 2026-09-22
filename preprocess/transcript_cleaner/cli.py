from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .gemini import GeminiCleaner, GeminiConfig
from .io import discover_inputs, read_jsonl
from .pipeline import RunSummary, TranscriptPipeline, preserve_by_code
from .rate_limit import SlidingWindowRateLimiter
from .state import StateStore
from .verify import verify_output
from .video_request import VideoRequestConfig, create_video_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean one JSONL file or a directory, one Gemini request per video."
    )
    parser.add_argument(
        "input_path", type=Path, help="One source .jsonl file or an input directory"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("clean_transcript"))
    parser.add_argument(
        "--state-dir", type=Path, default=Path(".clean_transcript_state")
    )
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument(
        "--payload-format",
        choices=("json-segments", "marker-text"),
        default="json-segments",
    )
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=65_536)
    parser.add_argument("--max-input-tokens", type=int, default=100_000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--large-request-threshold", type=int, default=10_000)
    parser.add_argument("--large-request-concurrency", type=int, default=2)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument(
        "--limit", type=int, default=0, help="Process first N files; 0 means all"
    )
    parser.add_argument(
        "--rpm", type=int, default=0, help="0 disables proactive RPM limiting"
    )
    parser.add_argument(
        "--tpm", type=int, default=0, help="0 disables proactive input-TPM limiting"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--watch-sentinel", type=Path,
        help="Watch an input directory for newly collected JSONL until this file exists.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument(
        "--dry-run", action="store_true", help="Inspect request sizes without Gemini"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify existing outputs"
    )
    return parser


def _resolve_sources(args: argparse.Namespace, *, allow_empty: bool = False) -> list[Path]:
    path = args.input_path
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError("Input file must have the .jsonl suffix")
        sources = [path]
    elif path.is_dir():
        sources = discover_inputs(path, args.pattern)
    else:
        raise ValueError(f"Input path does not exist: {path}")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if args.limit:
        sources = sources[: args.limit]
    if not sources and not allow_empty:
        raise ValueError(f"No input JSONL files found under {path}")
    return sources


def _request_config(args: argparse.Namespace) -> VideoRequestConfig:
    return VideoRequestConfig(max_estimated_input_tokens=args.max_input_tokens)


def _request_stats(
    path: Path, config: VideoRequestConfig, payload_format: str
) -> dict[str, int | str]:
    segments = read_jsonl(path)
    cleanable = (
        list(segments)
        if payload_format == "marker-text"
        else [
            segment
            for segment in segments
            if not preserve_by_code(segment.text)
        ]
    )
    request = create_video_request(cleanable, config) if cleanable else None
    return {
        "input": str(path),
        "segments": len(segments),
        "segments_sent_to_gemini": len(cleanable),
        "markers_preserved_by_code": len(segments) - len(cleanable),
        "estimated_input_tokens": request.estimated_input_tokens if request else 0,
        "gemini_requests": 1 if request else 0,
    }


def _dry_run_report(
    args: argparse.Namespace, sources: list[Path]
) -> dict[str, int | str]:
    stats = [
        _request_stats(source, _request_config(args), args.payload_format)
        for source in sources
    ]
    estimates = sorted(int(item["estimated_input_tokens"]) for item in stats)
    return {
        "files": len(stats),
        "segments": sum(int(item["segments"]) for item in stats),
        "gemini_requests": sum(int(item["gemini_requests"]) for item in stats),
        "estimated_input_tokens_total": sum(estimates),
        "estimated_input_tokens_p50": estimates[len(estimates) // 2],
        "estimated_input_tokens_max": max(estimates),
        "payload_format": args.payload_format,
        "concurrency": args.concurrency,
    }


def _verify_report(
    sources: list[Path], output_dir: Path, *, require_all: bool
) -> dict[str, int]:
    verified = 0
    segments = 0
    changed = 0
    missing = 0
    for source in sources:
        output = output_dir / source.name
        if not output.exists():
            missing += 1
            if require_all:
                raise ValueError(f"Missing output: {output}")
            continue
        report = verify_output(source, output)
        verified += 1
        segments += report.segment_count
        changed += report.changed_text_count
    return {
        "verified_output_files": verified,
        "verified_segments": segments,
        "changed_text_segments": changed,
        "missing_output_files": missing,
    }


async def _execute(
    args: argparse.Namespace, api_key: str, sources: list[Path]
) -> int:
    request_config = _request_config(args)
    gemini_config = GeminiConfig(
        model=args.model,
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
        max_output_tokens=args.max_output_tokens,
        payload_format=args.payload_format,
    )
    generation_settings = {
        "temperature": gemini_config.temperature,
        "max_output_tokens": gemini_config.max_output_tokens,
    }
    if gemini_config.payload_format != "json-segments":
        generation_settings["payload_format"] = gemini_config.payload_format
    limiter = SlidingWindowRateLimiter(args.rpm, args.tpm)
    state = StateStore(args.state_dir)

    async with GeminiCleaner(
        api_key=api_key,
        config=gemini_config,
        rate_limiter=limiter,
    ) as cleaner:
        def new_pipeline() -> TranscriptPipeline:
            return TranscriptPipeline(
                cleaner=cleaner,
                state_store=state,
                output_dir=args.output_dir,
                request_config=request_config,
                model=args.model,
                concurrency=args.concurrency,
                overwrite=args.overwrite,
                generation_settings=generation_settings,
                include_non_speech_markers=args.payload_format == "marker-text",
                large_request_threshold=args.large_request_threshold,
                large_request_concurrency=args.large_request_concurrency,
            )

        with state.exclusive_run():
            if args.watch_sentinel is None:
                summary = await new_pipeline().run(sources)
            else:
                seen = set(sources)
                all_sources = list(sources)
                summaries: list[RunSummary] = []
                if sources:
                    summaries.append(await new_pipeline().run(sources))
                while not args.watch_sentinel.is_file():
                    current = _resolve_sources(args, allow_empty=True)
                    pending = [source for source in current if source not in seen]
                    if pending:
                        seen.update(pending)
                        all_sources.extend(pending)
                        batch_summary = await new_pipeline().run(pending)
                        summaries.append(batch_summary)
                        if batch_summary.stopped_early:
                            break
                    else:
                        await asyncio.sleep(args.poll_interval)
                current = _resolve_sources(args, allow_empty=True)
                final_batch = [source for source in current if source not in seen]
                if final_batch:
                    all_sources.extend(final_batch)
                    summaries.append(await new_pipeline().run(final_batch))
                sources = all_sources
                summary = RunSummary(
                    discovered_files=sum(item.discovered_files for item in summaries),
                    completed_files=sum(item.completed_files for item in summaries),
                    skipped_files=sum(item.skipped_files for item in summaries),
                    failed_files=sum(item.failed_files for item in summaries),
                    api_chunks=sum(item.api_chunks for item in summaries),
                    started_api_work_items=sum(item.started_api_work_items for item in summaries),
                    resumed_chunks=sum(item.resumed_chunks for item in summaries),
                    stopped_early=any(item.stopped_early for item in summaries),
                    stop_reason=next((item.stop_reason for item in summaries if item.stop_reason), None),
                )

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print(
        json.dumps(
            _verify_report(sources, args.output_dir, require_all=False),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if summary.failed_files else 0


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.concurrency < 1:
            raise ValueError("--concurrency must be positive")
        if args.large_request_threshold < 1:
            raise ValueError("--large-request-threshold must be positive")
        if args.large_request_concurrency < 1:
            raise ValueError("--large-request-concurrency must be positive")
        if args.poll_interval <= 0:
            raise ValueError("--poll-interval must be positive")
        if args.watch_sentinel is not None and (args.dry_run or args.verify_only):
            raise ValueError("--watch-sentinel cannot be combined with dry-run/verify-only")
        sources = _resolve_sources(args, allow_empty=args.watch_sentinel is not None)
        if args.verify_only:
            print(
                json.dumps(
                    _verify_report(sources, args.output_dir, require_all=True),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_report(args, sources), ensure_ascii=False, indent=2
                )
            )
            return

        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {args.api_key_env} is not set. "
                "Set it locally; do not put the API key in a command or source file."
            )
        raise SystemExit(asyncio.run(_execute(args, api_key, sources)))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
