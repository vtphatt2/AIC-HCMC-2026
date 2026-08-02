"""Inspect video properties through the same extractor used by preprocessing."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.keyframes.contracts import VideoSource
from preprocess.keyframes.extractors.ffmpeg import FFmpegKeyframeExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description="Print measured properties of a video file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    source = VideoSource(args.video_id, args.input.expanduser().resolve())
    print(json.dumps(asdict(FFmpegKeyframeExtractor().probe(source)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
