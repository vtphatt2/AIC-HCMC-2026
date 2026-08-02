"""Select and render MP4 keyframes using explicit caller-provided paths.

Example:
    python preprocess/extract_keyframes.py \
      --input /path/to/video.mp4 --video-id VIDEO_001 \
      --output-root /path/to/keyframes-rendered
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support direct ``python preprocess/extract_keyframes.py`` invocation from any
# working directory without requiring a repository-wide packaging migration.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.keyframes.contracts import ExtractionOutput, RenderProfile, SceneSegment, VideoSource
from preprocess.keyframes.extractors.ffmpeg import FFmpegKeyframeExtractor
from preprocess.keyframes.manifest import write_selection_manifest
from preprocess.keyframes.selectors.scene_segments import SceneSegmentSelector
from preprocess.keyframes.selectors.uniform import UniformIntervalSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and render video keyframes.")
    parser.add_argument("--input", type=Path, required=True, help="Source MP4/video path.")
    parser.add_argument("--video-id", required=True, help="Stable ID used in output and manifests.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root for manifests and rendered images.")
    parser.add_argument("--selector", choices=("uniform", "scene-segments"), default="uniform")
    parser.add_argument("--interval-seconds", type=float, default=1.0, help="Uniform selection interval (default: 1.0).")
    parser.add_argument(
        "--scene-segments",
        type=Path,
        help="JSON file with {'segments': [{'start_ms': int, 'end_ms': int}, ...]}; required for scene-segments.",
    )
    parser.add_argument(
        "--short-edge",
        type=int,
        default=None,
        help="Output short edge in pixels; default preserves decoded dimensions.",
    )
    parser.add_argument("--image-format", choices=("png", "jpeg"), default="png", help="Output format (default: png).")
    parser.add_argument("--jpeg-quality", type=int, default=100, help="JPEG quality from 1 to 100 (default: 100).")
    parser.add_argument("--png-compress-level", type=int, default=6, help="PNG compression level 0 to 9 (default: 6).")
    parser.add_argument("--profile-id", help="Output profile ID; defaults from rendering options.")
    parser.add_argument("--allow-upscale", action="store_true", help="Allow resize to enlarge smaller source images.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing rendered images for this profile.")
    parser.add_argument("--dry-run", action="store_true", help="Probe and select only; write no manifests or images.")
    return parser.parse_args()


def load_scene_segments(path: Path) -> list[SceneSegment]:
    """Read a small, detector-agnostic scene manifest into validated intervals."""
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Scene segment file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene segment JSON is invalid: {path}: {exc}") from exc
    records = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Scene segment JSON must be a list or an object with a 'segments' list")
    return [SceneSegment(start_ms=int(item["start_ms"]), end_ms=int(item["end_ms"])) for item in records]


def main() -> int:
    args = parse_args()
    interval_ms = round(args.interval_seconds * 1000)
    if interval_ms <= 0:
        raise ValueError("--interval-seconds must be positive")

    if args.selector == "scene-segments" and args.scene_segments is None:
        raise ValueError("--scene-segments is required when --selector scene-segments")
    if args.selector == "uniform" and args.scene_segments is not None:
        raise ValueError("--scene-segments can only be used with --selector scene-segments")

    profile_suffix = "png" if args.image_format == "png" else f"jpeg-q{args.jpeg_quality}"
    profile_prefix = "original" if args.short_edge is None else f"short-{args.short_edge}"
    profile_id = args.profile_id or f"{profile_prefix}-{profile_suffix}"
    profile = RenderProfile(
        profile_id=profile_id,
        target_short_edge_px=args.short_edge,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        png_compress_level=args.png_compress_level,
        allow_upscale=args.allow_upscale,
    )
    source = VideoSource(video_id=args.video_id, path=args.input.expanduser().resolve())
    extractor = FFmpegKeyframeExtractor()
    selector = (
        UniformIntervalSelector(interval_ms=interval_ms)
        if args.selector == "uniform"
        else SceneSegmentSelector(load_scene_segments(args.scene_segments))
    )

    video = extractor.probe(source)
    selected = selector.select(video, extractor.scan(source, video))
    print(f"Selected {len(selected)} frames from {source.video_id} ({video.duration_ms / 1000:.3f}s).")
    if args.dry_run:
        return 0

    selection_manifest = args.output_root / "selection-manifests" / f"{source.video_id}.json"
    write_selection_manifest(
        selection_manifest, source, video, selector.name, selector.version,
        (
            {"interval_ms": interval_ms}
            if args.selector == "uniform"
            else {"scene_segments_path": str(args.scene_segments.expanduser().resolve())}
        ),
        selected,
    )
    result = extractor.materialize(
        source,
        video,
        selected,
        ExtractionOutput(
            rendered_root=args.output_root,
            selection_manifest_path=selection_manifest,
            profile=profile,
            overwrite=args.overwrite,
        ),
    )
    print(
        f"Rendered {len(result.written)} frames; skipped {len(result.skipped)}; "
        f"manifest: {result.rendered_manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
