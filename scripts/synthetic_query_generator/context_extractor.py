import json
import base64
import logging
from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError

from . import config

logger = logging.getLogger(__name__)


def _load_scenes(video_id: str) -> list[tuple[int, int]]:
    """Parse scenes.txt → list of (start_frame, end_frame)."""
    path = config.SCENES_ROOT / f"{video_id}.mp4.scenes.txt"
    if not path.exists():
        return []
    scenes = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            scenes.append((int(parts[0]), int(parts[1])))
    return scenes


def _load_keyframes(video_id: str) -> list[int]:
    path = config.SELECTED_KEYFRAMES_ROOT / f"{video_id}.txt"
    if not path.exists():
        return []
    return [int(l.strip()) for l in path.read_text().strip().splitlines() if l.strip()]


def _load_transcripts(video_id: str) -> list[dict]:
    path = config.TRANSCRIPTS_ROOT / f"{video_id}.jsonl"
    if not path.exists():
        return []
    transcripts = []
    for line in path.read_text().strip().splitlines():
        if line.strip():
            transcripts.append(json.loads(line))
    return transcripts


def _keyframes_in_scene(keyframes: list[int], scene_start: int, scene_end: int) -> list[int]:
    return [kf for kf in keyframes if scene_start <= kf <= scene_end]


def _select_representative(keyframe_ids: list[int], max_n: int) -> list[int]:
    """Pick evenly spaced keyframes, capped at max_n."""
    if len(keyframe_ids) <= max_n:
        return keyframe_ids
    step = max(1, len(keyframe_ids) // max_n)
    return [keyframe_ids[i] for i in range(0, len(keyframe_ids), step)][:max_n]


def _frame_to_ms(frame_idx: int) -> int:
    return int(frame_idx / config.FPS * 1000)


def _extract_transcript_for_scene(
    transcripts: list[dict], start_ms: int, end_ms: int, window_ms: int
) -> str:
    """Get transcript text overlapping or within +- window of the scene."""
    window_start = start_ms - window_ms
    window_end = end_ms + window_ms
    relevant = []
    for seg in transcripts:
        if seg["end_time_ms"] < window_start or seg["start_time_ms"] > window_end:
            continue
        relevant.append(seg)
    if not relevant:
        return ""
    return " ".join(seg["text"] for seg in relevant)


def _image_to_base64(image_path: Path) -> str:
    """Load image, compress if over limit, encode as base64. Returns '' on failure."""
    try:
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return ""
        img = Image.open(image_path)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=config.IMAGE_QUALITY)
        size = buf.tell()
        if size > config.MAX_IMAGE_SIZE_BYTES:
            logger.warning(
                f"Image too large ({size} bytes), compressing further: {image_path.name}"
            )
            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=50)
        encoded = base64.b64encode(buf.getvalue()).decode()
        return encoded
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Corrupt/unreadable image skipped: {image_path} — {e}")
        return ""


def extract_scene_contexts(video_id: str) -> list[dict]:
    """Main extraction: scenes → keyframes + transcripts + images."""
    scenes = _load_scenes(video_id)
    keyframes = _load_keyframes(video_id)
    transcripts = _load_transcripts(video_id)

    if not scenes or not keyframes:
        return []

    window_ms = int(config.TRANSCRIPT_WINDOW_SEC * 1000)
    image_dir = config.USED_KEYFRAMES_ROOT / video_id
    contexts = []

    for scene_start, scene_end in scenes:
        kfs = _keyframes_in_scene(keyframes, scene_start, scene_end)
        if not kfs:
            continue

        reps = _select_representative(kfs, config.MAX_KEYFRAMES_PER_SCENE)

        start_ms = _frame_to_ms(scene_start)
        end_ms = _frame_to_ms(scene_end)
        transcript_text = _extract_transcript_for_scene(transcripts, start_ms, end_ms, window_ms)

        images_b64 = []
        for kf in reps:
            img_path = image_dir / f"{kf:06d}.jpg"
            b64 = _image_to_base64(img_path)
            if b64:
                images_b64.append(b64)

        if not images_b64:
            logger.warning(f"Scene {scene_start}-{scene_end}: no valid images, skipping")
            continue

        contexts.append({
            "video_id": video_id,
            "scene_start_frame": scene_start,
            "scene_end_frame": scene_end,
            "timestamp_range": [round(start_ms / 1000, 1), round(end_ms / 1000, 1)],
            "keyframe_ids": [f"{kf:06d}.jpg" for kf in reps],
            "transcript_text": transcript_text,
            "images_base64": images_b64,
        })

    return contexts


# ponytail: O(n*m) scan per scene over transcripts; fine for 9 videos.
# If scaling to thousands, index by time-range with bisect.
