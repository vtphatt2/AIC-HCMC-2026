"""Render a transcript banner for every keyframe using the local sentence index.

Input layout:

  AIC2026_sample/
  ├── keyframes/<video_id>/<frame_id>.jpg  # or keyframes_org/
  └── keyframe_transcript_index/<video_id>.json

The index is sentence-centric. For each keyframe, the script derives a
timestamp from its six-digit filename and finds the sentence interval that
contains it. Frames outside every sentence interval still receive a blank
banner. No transcript windowing or raw TXT parsing occurs in this script.

Example:

  preprocess/.venv/bin/python -m \
      preprocess.transcript.add_transcript_banner --video-id L03_V002
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_ROOT = PROJECT_ROOT / "AIC2026_sample"
TRANSCRIPT_PREFIX = "Audio content (in Vietnamese): "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render banners from keyframe transcript indexes.")
    parser.add_argument("--video-id", required=True, help="Dataset video ID, for example L03_V002.")
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument(
        "--data-root",
        dest="sample_root",
        type=Path,
        help="Alias for --sample-root, useful for the SSH batch data layout.",
    )
    parser.set_defaults(sample_root=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: <sample-root>/subtitled_keyframes.")
    parser.add_argument("--image-extension", default=".jpg", help="Keyframe file extension.")
    parser.add_argument("--font-path", type=Path, help="Unicode TTF/OTF font path.")
    parser.add_argument("--font-size", type=int, default=30)
    parser.add_argument("--panel-height", type=int, default=200, help="Minimum panel height in pixels.")
    parser.add_argument("--overflow", choices=("expand", "truncate", "error"), default="error")
    parser.add_argument("--existing", choices=("overwrite", "skip", "unique"), default="overwrite")
    parser.add_argument("--dry-run", action="store_true", help="Validate index and input images without writing output.")
    return parser.parse_args()


def load_index(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Keyframe transcript index not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sentences"), list):
        raise ValueError(f"Invalid sentence index: {path}")
    return payload


def load_font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont:
    if size <= 0:
        raise ValueError("--font-size must be positive")
    candidates = [font_path] if font_path else []
    candidates.extend([
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ])
    for candidate in candidates:
        if candidate and candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError("No Unicode font found. Pass --font-path to a .ttf/.otf font.")


def wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def panel_lines(lines: list[str], font: ImageFont.FreeTypeFont, requested_height: int, overflow: str, padding: int, spacing: int) -> tuple[list[str], int]:
    if requested_height <= 0:
        raise ValueError("--panel-height must be positive")
    line_height = sum(font.getmetrics())
    required_height = padding * 2 + len(lines) * line_height + max(0, len(lines) - 1) * spacing
    if required_height <= requested_height:
        return lines, requested_height
    if overflow == "expand":
        return lines, required_height
    if overflow == "error":
        raise ValueError(f"Transcript needs {required_height}px but --panel-height is {requested_height}px")
    max_lines = max(1, (requested_height - 2 * padding + spacing) // (line_height + spacing))
    return lines[:max_lines], requested_height


def output_path(output_dir: Path, source_path: Path, policy: str) -> Path | None:
    candidate = output_dir / source_path.name
    if not candidate.exists() or policy == "overwrite":
        return candidate
    if policy == "skip":
        return None
    for number in range(1, 10_000):
        unique = output_dir / f"{source_path.stem}_{number}{source_path.suffix}"
        if not unique.exists():
            return unique
    raise RuntimeError("Unable to allocate a unique output filename")


def render_banner(image_path: Path, text: str, font: ImageFont.FreeTypeFont, panel_height: int, overflow: str) -> Image.Image:
    padding, spacing = 10, 5
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    probe = ImageDraw.Draw(Image.new("RGB", (image.width, 1), "white"))
    lines = wrap_text(text, probe, font, image.width - 2 * padding)
    lines, resolved_height = panel_lines(lines, font, panel_height, overflow, padding, spacing)

    result = Image.new("RGB", (image.width, image.height + resolved_height), "white")
    result.paste(image, (0, 0))
    draw = ImageDraw.Draw(result)
    y = image.height + padding
    line_height = sum(font.getmetrics())
    for line in lines:
        draw.text((padding, y), line, font=font, fill="black")
        y += line_height + spacing
    return result


def frame_id(path: Path) -> str:
    groups = re.findall(r"\d+", path.stem)
    if not groups:
        raise ValueError(f"Invalid keyframe filename: {path.name}")
    return groups[-1].zfill(6)


def collect_keyframes(directory: Path, extension: str) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Keyframe directory not found: {directory}")
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == extension.lower()]
    by_frame_id: dict[str, Path] = {}
    for path in paths:
        current_id = frame_id(path)
        existing = by_frame_id.get(current_id)
        canonical_name = f"{current_id}{extension.lower()}"
        if existing is None or (path.name == canonical_name and existing.name != canonical_name):
            by_frame_id[current_id] = path
    return sorted(by_frame_id.values(), key=lambda path: (int(frame_id(path)), path.name))


def resolve_keyframe_dir(sample_root: Path, video_id: str) -> Path:
    """Support both the standard and locally renamed keyframe directories."""
    for root_name in ("keyframes", "keyframes_org"):
        candidate = sample_root / root_name / video_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Keyframe directory not found for {video_id}; expected "
        f"{sample_root / 'keyframes' / video_id} or "
        f"{sample_root / 'keyframes_org' / video_id}"
    )


def sentence_for_timestamp(sentences: list[dict], timestamp_ms: int) -> dict | None:
    for sentence in sentences:
        if int(sentence["start_ms"]) <= timestamp_ms < int(sentence["end_ms"]):
            return sentence
    return None


def load_frame_timestamps(sample_root: Path, video_id: str) -> dict[str, int]:
    for root_name in ("keyframes", "keyframes_org"):
        manifest = sample_root / root_name / video_id / "manifest.json"
        if not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        timestamps: dict[str, int] = {}
        for frame in payload.get("frames", []):
            if not isinstance(frame, dict):
                continue
            raw = frame.get("frame", {})
            frame_number = str(raw.get("frame_id") or raw.get("source_frame_number", ""))
            if frame_number.isdigit() and raw.get("timestamp_ms") is not None:
                timestamps[frame_number] = int(raw["timestamp_ms"])
        return timestamps
    return {}


def optional_fps(payload: object) -> float | None:
    """Return a usable positive FPS value, treating missing/invalid values as unavailable."""
    if not isinstance(payload, dict):
        return None
    raw_fps = payload.get("fps")
    if raw_fps in (None, ""):
        return None
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def main() -> int:
    args = parse_args()
    sample_root = args.sample_root.resolve()
    payload = load_index(sample_root / "keyframe_transcript_index" / f"{args.video_id}.json")
    if payload.get("video_id") != args.video_id:
        raise ValueError(f"Index video_id does not match --video-id: {payload.get('video_id')}")

    extension = args.image_extension if args.image_extension.startswith(".") else f".{args.image_extension}"
    keyframe_dir = resolve_keyframe_dir(sample_root, args.video_id)
    output_dir = (args.output_dir or sample_root / "subtitled_keyframes") / args.video_id
    fps = optional_fps(payload)
    frame_timestamps = load_frame_timestamps(sample_root, args.video_id)
    if fps is None and not frame_timestamps:
        raise ValueError("Index has no FPS and rendered keyframe PTS are unavailable")
    keyframes = collect_keyframes(keyframe_dir, extension)
    font = load_font(args.font_path, args.font_size)

    print(f"video={args.video_id} fps={fps} timing_source={payload.get('timing_source')} keyframes={len(keyframes)} sentences={len(payload['sentences'])}")
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0
    with_transcript = 0
    for index, image_path in enumerate(keyframes, start=1):
        current_frame_id = frame_id(image_path)
        timestamp_ms = frame_timestamps.get(
            current_frame_id,
            int(int(current_frame_id) / fps * 1000) if fps is not None else -1,
        )
        sentence = sentence_for_timestamp(payload["sentences"], timestamp_ms)
        text = f"{TRANSCRIPT_PREFIX}{sentence['text']}" if sentence is not None else ""
        destination = output_path(output_dir, image_path, args.existing)
        if destination is None:
            skipped += 1
            continue
        image = render_banner(image_path, text, font, args.panel_height, args.overflow)
        image.save(destination)
        saved += 1
        with_transcript += sentence is not None
        status = sentence["sentence_id"] if sentence is not None else "no-transcript"
        print(f"[{index}/{len(keyframes)}] {current_frame_id} -> {destination.name} ({status})")

    print(f"Done: saved={saved}, skipped={skipped}, with_transcript={with_transcript}, output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
