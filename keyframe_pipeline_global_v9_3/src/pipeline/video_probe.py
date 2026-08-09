"""ffprobe wrapper shared by the TransNet and embedding phases."""
from __future__ import annotations

import json

from .zip_source import run_checked


def parse_ratio(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    numerator, separator, denominator = value.partition("/")
    if separator:
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    return float(numerator)


def probe_video(ffprobe_bin: str, video_source: str) -> tuple[float, int, int]:
    result = run_checked(
        [
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate,width,height",
            "-of", "json", video_source,
        ],
        capture_stdout=True,
    )
    streams = json.loads(result.stdout.decode("utf-8")).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream: {video_source}")
    stream = streams[0]
    fps = parse_ratio(stream.get("avg_frame_rate")) or parse_ratio(stream.get("r_frame_rate"))
    width, height = int(stream["width"]), int(stream["height"])
    if not fps or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: fps={fps}, size={width}x{height}")
    return fps, width, height
