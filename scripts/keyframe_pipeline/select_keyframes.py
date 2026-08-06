"""
Local (cheap, no GPU/decode) step: turn TransNetV2 scenes.json into a flat
list of keyframe frame indices, using the duration-based ladder from
../../../AIC-HCMC-2026/docs/keyframe_selection.md:

  scene duration <= 3s   -> 1 keyframe  (sampled at 50% of the scene)
  scene duration <= 10s  -> 3 keyframes (at 1/6, 3/6, 5/6 of the scene)
  scene duration > 10s   -> 5 keyframes (at 1/10, 3/10, .., 9/10 of the scene)

Usage:
  python select_keyframes.py --scenes SCENES/scenes.json --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, required=True, help="scenes.json from colab_run_transnet.py")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def frame_count_for(duration_s: float) -> int:
    if duration_s <= 3:
        return 1
    if duration_s <= 10:
        return 3
    return 5


def select_scene(start_frame: int, end_frame: int, fps: float) -> list[int]:
    duration_frames = end_frame - start_frame + 1
    duration_s = duration_frames / fps
    count = frame_count_for(duration_s)
    frames = {
        min(end_frame, max(start_frame, start_frame + round((i + 0.5) * duration_frames / count)))
        for i in range(count)
    }
    return sorted(frames)


def main() -> None:
    args = parse_args()
    data = json.loads(args.scenes.read_text())
    fps = data["fps"]

    keyframes: list[dict] = []
    for scene_index, scene in enumerate(data["scenes"]):
        for frame_number in select_scene(scene["start_frame"], scene["end_frame"], fps):
            keyframes.append({"frame_number": frame_number, "scene_index": scene_index})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "video": data["video"],
        "fps": fps,
        "num_scenes": len(data["scenes"]),
        "num_keyframes": len(keyframes),
        "keyframes": keyframes,
    }
    (args.out_dir / "keyframes.json").write_text(json.dumps(result, indent=2))
    print(f"{len(data['scenes'])} scenes -> {len(keyframes)} keyframes -> {args.out_dir / 'keyframes.json'}")


if __name__ == "__main__":
    main()
