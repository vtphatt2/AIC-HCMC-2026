"""Primary command line interface for ZIP-native preprocessing."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .run_config import apply_run_config

from .zip_pipeline import (
    ResultArchiveValidator,
    ZipPipelineConfig,
    build_engine_command,
    run_pipeline,
    shell_join,
)
from .transcript_stage import (
    CollectionCleaningConfig,
    TranscriptStageConfig,
    build_collection_cleaning_command,
    build_transcript_command,
    lot_video_id_regex,
    run_commands_concurrently,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m preprocess",
        description="Decode and embed videos directly from a source ZIP.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run ZIP-native TransNet + PE-Core preprocessing.")
    source = run.add_mutually_exclusive_group(required=False)
    source.add_argument("--url", help="Download one complete source ZIP, then process it in place.")
    source.add_argument("--zip", dest="zip_path", type=Path, help="Use an existing local source ZIP.")
    run.add_argument("--config", type=Path, help="JSON run definition; supplies source and all run settings.")
    run.add_argument("--work-root", type=Path, default=Path("data/zip-preprocess"))
    run.add_argument("--archive", type=Path, help="Final challenge-compatible *_results.zip path.")
    run.add_argument("--profile", choices=("balanced", "ram-rich", "disk-rich", "colab"), default="balanced")
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--prefetch-batches", type=int, default=3)
    run.add_argument("--transnet-batch-size", type=int, default=16)
    run.add_argument("--transnet-decode-workers", type=int, default=4)
    run.add_argument("--transnet-prefetch-windows", type=int, default=256)
    run.add_argument(
        "--keyframe-strategy", choices=("tiered", "linear"), default="tiered",
        help="Per-scene keyframe count policy (default: tiered).",
    )
    run.add_argument("--keyframes-per-second", type=float, default=0.3)
    run.add_argument("--min-keyframes-per-scene", type=int, default=1)
    run.add_argument("--max-keyframes-per-scene", type=int, default=20)
    run.add_argument("--limit", type=int)
    run.add_argument("--sequential-stages", action="store_true", help="Disable per-video stage streaming.")
    run.add_argument("--delete-source-before-package", action="store_true")
    run.add_argument("--local-files-only", action="store_true")
    transcript_source = run.add_mutually_exclusive_group()
    transcript_source.add_argument(
        "--transcripts", type=Path,
        help="Clean an existing directory/file of transcript JSONL in parallel with video preprocessing.",
    )
    transcript_source.add_argument(
        "--transcript-metadata", type=Path,
        help="Collect captions from media-info JSON/ZIP, then clean them while video preprocessing runs.",
    )
    run.add_argument("--transcript-raw-dir", type=Path)
    run.add_argument("--transcript-output-dir", type=Path)
    run.add_argument("--transcript-state-dir", type=Path)
    run.add_argument("--transcript-model", default="gemini-3.5-flash-lite")
    run.add_argument("--transcript-concurrency", type=int, default=8)
    run.add_argument("--transcript-rpm", type=int, default=0)
    run.add_argument("--transcript-tpm", type=int, default=0)
    run.add_argument(
        "--transcript-video-id-regex",
        help="Metadata video filter; defaults to the source lot prefix, e.g. ^L30_.",
    )
    run.add_argument("--transcript-overwrite", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Print the engine command without executing it.")

    inspect = commands.add_parser("inspect", help="Validate a generated challenge result ZIP.")
    inspect.add_argument("archive", type=Path)
    commands.add_parser("clean-transcripts", add_help=False, help="Clean existing transcript JSONL.")
    commands.add_parser("collect-transcripts", add_help=False, help="Collect captions from media-info.")
    commands.add_parser("transcripts", add_help=False, help="Collection/normalization/cleaning workflow.")
    return parser


def _config(args: argparse.Namespace) -> ZipPipelineConfig:
    return ZipPipelineConfig(
        source_url=args.url,
        source_zip=args.zip_path,
        work_root=args.work_root,
        archive=args.archive,
        profile=args.profile,
        device=args.device,
        batch_size=args.batch_size,
        prefetch_batches=args.prefetch_batches,
        transnet_batch_size=args.transnet_batch_size,
        transnet_decode_workers=args.transnet_decode_workers,
        transnet_prefetch_windows=args.transnet_prefetch_windows,
        keyframe_strategy=args.keyframe_strategy,
        keyframes_per_second=args.keyframes_per_second,
        min_keyframes_per_scene=args.min_keyframes_per_scene,
        max_keyframes_per_scene=args.max_keyframes_per_scene,
        limit=args.limit,
        parallel_stages=not args.sequential_stages,
        delete_source_before_package=args.delete_source_before_package,
        local_files_only=args.local_files_only,
    )


def _transcript_command(args: argparse.Namespace, video_config: ZipPipelineConfig) -> list[str] | None:
    output_dir = args.transcript_output_dir or args.work_root / "clean_transcript"
    state_dir = args.transcript_state_dir or args.work_root / ".clean_transcript_state"
    if args.transcript_metadata is not None:
        raw_dir = args.transcript_raw_dir or args.work_root / "transcripts_raw" / video_config.data_id
        video_id_regex = args.transcript_video_id_regex
        if video_id_regex is None:
            video_id_regex = lot_video_id_regex(video_config.data_id)
        return build_collection_cleaning_command(CollectionCleaningConfig(
            metadata_path=args.transcript_metadata,
            raw_dir=raw_dir,
            output_dir=output_dir,
            state_dir=state_dir,
            collection_concurrency=args.transcript_concurrency,
            cleaning_concurrency=args.transcript_concurrency,
            model=args.transcript_model,
            rpm=args.transcript_rpm,
            tpm=args.transcript_tpm,
            video_id_regex=video_id_regex,
            overwrite=args.transcript_overwrite,
        ))
    if args.transcripts is not None:
        return build_transcript_command(TranscriptStageConfig(
            input_path=args.transcripts,
            output_dir=output_dir,
            state_dir=state_dir,
            model=args.transcript_model,
            concurrency=args.transcript_concurrency,
            rpm=args.transcript_rpm,
            tpm=args.transcript_tpm,
            overwrite=args.transcript_overwrite,
        ))
    return None


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in {"clean-transcripts", "collect-transcripts", "transcripts"}:
        command = raw_args[0]
        if command == "clean-transcripts":
            target = "preprocess.transcript_cleaner.cli"
            forwarded = raw_args[1:]
        else:
            target = "preprocess.transcript_cleaner.workflow"
            forwarded = (["collect"] if command == "collect-transcripts" else []) + raw_args[1:]
        return subprocess.run([sys.executable, "-m", target, *forwarded], check=False).returncode

    args = build_parser().parse_args(raw_args)
    try:
        if args.command == "inspect":
            inspection = ResultArchiveValidator().validate(args.archive)
            print(json.dumps(inspection.to_dict(), indent=2))
            return 0

        args = apply_run_config(args)
        config = _config(args)
        transcript_command = _transcript_command(args, config)
        if args.dry_run:
            print(shell_join(build_engine_command(config)))
            if transcript_command is not None:
                print(shell_join(transcript_command))
            return 0
        if transcript_command is None:
            inspection = run_pipeline(config)
        else:
            statuses = run_commands_concurrently({
                "video": build_engine_command(config),
                "transcript": transcript_command,
            })
            failures = {name: code for name, code in statuses.items() if code != 0}
            if failures:
                raise RuntimeError(f"preprocessing stage failures: {failures}")
            inspection = ResultArchiveValidator().validate(config.result_archive)
            if inspection.format_version != 3:
                raise ValueError("new pipeline output must use manifest format_version 3")
        print(json.dumps(inspection.to_dict(), indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
