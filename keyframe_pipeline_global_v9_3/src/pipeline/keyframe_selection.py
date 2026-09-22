"""Scene-aware keyframe selection policies shared by every TransNet launcher."""
from __future__ import annotations

import math
import json
from collections.abc import Iterable
from pathlib import Path


KEYFRAME_STRATEGIES = ("tiered", "linear")


def selection_metadata(
    *, strategy: str, keyframes_per_second: float,
    min_keyframes_per_scene: int, max_keyframes_per_scene: int,
) -> dict:
    return {
        "strategy": strategy,
        "keyframes_per_second": keyframes_per_second,
        "min_keyframes_per_scene": min_keyframes_per_scene,
        "max_keyframes_per_scene": max_keyframes_per_scene,
    }


def _selection_matches(payload: object, expected: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    actual = payload.get("selection")
    # Artifacts created before selection metadata existed used the exact tiered
    # policy, so they remain valid for the backward-compatible default.
    if actual is None:
        return expected["strategy"] == "tiered"
    if not isinstance(actual, dict) or actual.get("strategy") != expected["strategy"]:
        return False
    if expected["strategy"] == "tiered":
        return True
    return all(actual.get(key) == expected[key] for key in (
        "keyframes_per_second", "min_keyframes_per_scene", "max_keyframes_per_scene",
    ))


def invalidate_mismatched_selection(
    out_dir: Path,
    *, strategy: str, keyframes_per_second: float,
    min_keyframes_per_scene: int, max_keyframes_per_scene: int,
) -> list[str]:
    """Remove derived artifacts whose keyframe policy differs from this run."""
    expected = selection_metadata(
        strategy=strategy,
        keyframes_per_second=keyframes_per_second,
        min_keyframes_per_scene=min_keyframes_per_scene,
        max_keyframes_per_scene=max_keyframes_per_scene,
    )
    invalidated: list[str] = []
    if not out_dir.is_dir():
        return invalidated
    for keyframes_path in sorted(out_dir.glob("*/keyframes.json")):
        try:
            payload = json.loads(keyframes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if _selection_matches(payload, expected):
            continue
        video_dir = keyframes_path.parent
        for name in (
            "keyframes.json", "embeddings.npy", "embeddings.partial.npy",
            "embedding_stats.json", "transnet_stats.json",
        ):
            (video_dir / name).unlink(missing_ok=True)
        invalidated.append(video_dir.name)
    return invalidated


def keyframe_count(
    duration_s: float,
    *,
    strategy: str = "tiered",
    keyframes_per_second: float = 0.3,
    min_keyframes_per_scene: int = 1,
    max_keyframes_per_scene: int = 20,
) -> int:
    """Return the requested sample count for one scene."""
    if strategy not in KEYFRAME_STRATEGIES:
        raise ValueError(f"unsupported keyframe strategy: {strategy}")
    if keyframes_per_second <= 0:
        raise ValueError("keyframes_per_second must be positive")
    if min_keyframes_per_scene < 1:
        raise ValueError("min_keyframes_per_scene must be positive")
    if max_keyframes_per_scene < min_keyframes_per_scene:
        raise ValueError("max_keyframes_per_scene must be >= min_keyframes_per_scene")

    if strategy == "tiered":
        if duration_s <= 3.0:
            return 1
        if duration_s <= 10.0:
            return 3
        return 5

    requested = math.ceil(max(0.0, duration_s) * keyframes_per_second)
    return min(max_keyframes_per_scene, max(min_keyframes_per_scene, requested))


def select_keyframes(
    scenes: Iterable,
    fps: float,
    *,
    strategy: str = "tiered",
    keyframes_per_second: float = 0.3,
    min_keyframes_per_scene: int = 1,
    max_keyframes_per_scene: int = 20,
) -> list[dict]:
    """Select evenly spaced frame centers inside every detected scene."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    selected: list[dict] = []
    for scene_index, (start_raw, end_raw) in enumerate(scenes):
        start_frame, end_frame = int(start_raw), int(end_raw)
        duration_frames = end_frame - start_frame + 1
        if duration_frames <= 0:
            continue
        count = min(duration_frames, keyframe_count(
            duration_frames / fps,
            strategy=strategy,
            keyframes_per_second=keyframes_per_second,
            min_keyframes_per_scene=min_keyframes_per_scene,
            max_keyframes_per_scene=max_keyframes_per_scene,
        ))
        numbers = sorted({
            min(end_frame, max(start_frame, start_frame + round((index + 0.5) * duration_frames / count)))
            for index in range(count)
        })
        selected.extend(
            {"frame_number": number, "scene_index": scene_index}
            for number in numbers
        )
    return selected
