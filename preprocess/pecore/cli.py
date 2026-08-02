"""Command line entry point for PE-Core visual feature generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from preprocess.pecore.embedding import (
    OpenClipPECoreEncoder,
    PECoreEmbeddingConfig,
    PECoreEmbeddingPipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed keyframe images with the full PE-Core image encoder."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing <video_id>/*.jpg.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root for <video_id>/*.npy.")
    parser.add_argument(
        "--video-id",
        action="append",
        help="Embed only this video directory; repeat for multiple videos. Default: all directories.",
    )
    parser.add_argument(
        "--model-id",
        default="hf-hub:timm/PE-Core-bigG-14-448",
        help="OpenCLIP/Hugging Face model identifier.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--expected-dim", type=int, default=1_280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing valid feature files instead of resuming them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PECoreEmbeddingConfig(
            enabled=True,
            model_id=args.model_id,
            device=args.device,
            precision=args.precision,
            expected_dim=args.expected_dim,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        encoder = OpenClipPECoreEncoder(config)
        pipeline = PECoreEmbeddingPipeline(
            encoder,
            batch_size=config.batch_size,
            image_extensions=config.image_extensions,
            overwrite=config.overwrite,
        )
        result = pipeline.embed_all(
            args.input_root,
            args.output_root,
            video_ids=args.video_id,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
