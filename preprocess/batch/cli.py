"""Command-line entry point for SSH execution."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from preprocess.batch.archive_validator import ZipArchiveValidator
from preprocess.batch.config import BatchConfig
from preprocess.batch.links import LinkListParser
from preprocess.batch.orchestrator import build_default_orchestrator
from preprocess.batch.preflight import PreflightChecker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the configurable SSH preprocessing pipeline.")
    parser.add_argument("--config", type=Path, help="JSON configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Check tools and disk space only.")
    preflight.set_defaults(handler=_run_preflight)

    links = subparsers.add_parser("parse-links", help="Print normalized archive requests.")
    links.set_defaults(handler=_parse_links)

    inspect = subparsers.add_parser("inspect-archive", help="Validate one downloaded ZIP.")
    inspect.add_argument("archive", type=Path)
    inspect.set_defaults(handler=_inspect_archive)

    run = subparsers.add_parser("run", help="Run all archive URLs through the pipeline.")
    run.add_argument("--links", type=Path, help="Override config.links_file.")
    run.add_argument("--data-root", type=Path, help="Override config.data_root.")
    run.add_argument("--metadata-root", type=Path, help="Override config.metadata_root.")
    run.set_defaults(handler=_run_pipeline)

    upload = subparsers.add_parser(
        "upload",
        help="Upload one already-processed lot without rerunning preprocessing.",
    )
    upload.add_argument("--lot-id", required=True, help="Lot directory to upload, e.g. L21_a.")
    upload.set_defaults(handler=_upload_lot)
    return parser


def load_config(args: argparse.Namespace) -> BatchConfig:
    config = BatchConfig.from_json(args.config) if args.config else BatchConfig()
    overrides = {}
    if getattr(args, "links", None) is not None:
        overrides["links_file"] = args.links
    if getattr(args, "data_root", None) is not None:
        overrides["data_root"] = args.data_root
    if getattr(args, "metadata_root", None) is not None:
        overrides["metadata_root"] = args.metadata_root
    return replace(config, **overrides) if overrides else config


def _run_preflight(args: argparse.Namespace) -> int:
    result = PreflightChecker(load_config(args)).run()
    print(json.dumps({"tools": result.tools, "free_bytes": result.free_bytes, "root": str(result.root)}, indent=2))
    return 0


def _parse_links(args: argparse.Namespace) -> int:
    config = load_config(args)
    requests = LinkListParser(
        archive_prefix=config.archive.prefix,
        archive_extension=config.archive.extension,
    ).parse(config.links_file)
    print(json.dumps([request.to_dict() for request in requests], ensure_ascii=False, indent=2))
    return 0


def _inspect_archive(args: argparse.Namespace) -> int:
    config = load_config(args)
    inspection = ZipArchiveValidator(config.archive).validate(args.archive)
    print(json.dumps(inspection.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    config = load_config(args)
    requests = LinkListParser(
        archive_prefix=config.archive.prefix,
        archive_extension=config.archive.extension,
    ).parse(config.links_file)
    results = build_default_orchestrator(config).run_all(requests)
    print(json.dumps({"completed_lots": [result.lot_id for result in results]}, ensure_ascii=False, indent=2))
    return 0


def _upload_lot(args: argparse.Namespace) -> int:
    config = load_config(args)
    result = build_default_orchestrator(config).upload_lot(args.lot_id)
    print(
        json.dumps(
            {"lot_id": args.lot_id, "upload": result.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
