"""Unified collection, legacy normalization, and Gemini-cleaning CLI."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

from .collect import CollectionConfig, collect_transcripts
from .normalize import normalize_txt_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect, normalize, and clean transcript JSONL files.")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="Collect YouTube captions from media-info JSON/ZIP.")
    collect.add_argument("metadata_path", type=Path)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--languages", nargs="+", default=["vi", "en"])
    collect.add_argument("--concurrency", type=int, default=8)
    collect.add_argument("--min-interval", type=float, default=0.5)
    collect.add_argument("--max-attempts", type=int, default=5)
    collect.add_argument("--video-id-regex")
    collect.add_argument("--overwrite", action="store_true")

    normalize = commands.add_parser("normalize", help="Convert legacy timestamped TXT to JSONL.")
    normalize.add_argument("input_dir", type=Path)
    normalize.add_argument("--output-dir", type=Path, required=True)
    normalize.add_argument("--overwrite", action="store_true")

    combined = commands.add_parser(
        "collect-clean", help="Collect captions and then safely clean all successful JSONL files."
    )
    combined.add_argument("metadata_path", type=Path)
    combined.add_argument("--raw-dir", type=Path, required=True)
    combined.add_argument("--output-dir", type=Path, required=True)
    combined.add_argument("--state-dir", type=Path, required=True)
    combined.add_argument("--languages", nargs="+", default=["vi", "en"])
    combined.add_argument("--collection-concurrency", type=int, default=8)
    combined.add_argument("--cleaning-concurrency", type=int, default=8)
    combined.add_argument("--min-interval", type=float, default=0.5)
    combined.add_argument("--max-attempts", type=int, default=5)
    combined.add_argument("--video-id-regex")
    combined.add_argument("--model", default="gemini-3.5-flash-lite")
    combined.add_argument("--rpm", type=int, default=0)
    combined.add_argument("--tpm", type=int, default=0)
    combined.add_argument("--overwrite", action="store_true")
    return parser


def _collection_config(args: argparse.Namespace, output_dir: Path) -> CollectionConfig:
    return CollectionConfig(
        metadata_path=args.metadata_path,
        output_dir=output_dir,
        languages=tuple(args.languages),
        concurrency=(
            args.collection_concurrency
            if hasattr(args, "collection_concurrency")
            else args.concurrency
        ),
        min_interval_seconds=args.min_interval,
        max_attempts=args.max_attempts,
        video_id_regex=args.video_id_regex,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "normalize":
            converted = normalize_txt_directory(args.input_dir, args.output_dir, overwrite=args.overwrite)
            print(json.dumps({"converted": converted, "output_dir": str(args.output_dir)}, indent=2))
            return 0

        if args.command == "collect":
            summary = collect_transcripts(_collection_config(args, args.output_dir))
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
            return 1 if summary.failed else 0

        clean_command = [
            sys.executable, "-m", "preprocess.transcript_cleaner.cli", str(args.raw_dir),
            "--output-dir", str(args.output_dir),
            "--state-dir", str(args.state_dir),
            "--model", args.model,
            "--concurrency", str(args.cleaning_concurrency),
            "--rpm", str(args.rpm),
            "--tpm", str(args.tpm),
        ]
        if args.overwrite:
            clean_command.append("--overwrite")
            summary = collect_transcripts(_collection_config(args, args.raw_dir))
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
            clean_result = subprocess.run(clean_command, check=False)
            return clean_result.returncode or (1 if summary.failed else 0)

        # In normal resumable mode, collection and cleaning form a filesystem-backed
        # producer/consumer pipeline. Start the watcher first; every atomically
        # published raw JSONL can enter Gemini while other captions are still fetched.
        sentinel = args.raw_dir / ".collection.complete"
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        sentinel.unlink(missing_ok=True)
        clean_command.extend(("--watch-sentinel", str(sentinel)))
        cleaner = subprocess.Popen(clean_command)
        clean_returncode = 1
        try:
            summary = collect_transcripts(_collection_config(args, args.raw_dir))
        finally:
            temporary = sentinel.with_suffix(".tmp")
            temporary.write_text("done\n", encoding="utf-8")
            os.replace(temporary, sentinel)
            clean_returncode = cleaner.wait()
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return clean_returncode or (1 if summary.failed else 0)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
