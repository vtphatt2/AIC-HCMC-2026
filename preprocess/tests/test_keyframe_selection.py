from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from keyframe_pipeline_global_v9_3.src.pipeline.keyframe_selection import (
    invalidate_mismatched_selection,
    keyframe_count,
    select_keyframes,
)


class KeyframeSelectionTest(unittest.TestCase):
    def test_tiered_strategy_preserves_existing_rules(self) -> None:
        self.assertEqual(keyframe_count(3.0, strategy="tiered"), 1)
        self.assertEqual(keyframe_count(3.1, strategy="tiered"), 3)
        self.assertEqual(keyframe_count(10.0, strategy="tiered"), 3)
        self.assertEqual(keyframe_count(10.1, strategy="tiered"), 5)

    def test_linear_strategy_scales_and_clamps_per_scene(self) -> None:
        options = {
            "strategy": "linear",
            "keyframes_per_second": 0.3,
            "min_keyframes_per_scene": 1,
            "max_keyframes_per_scene": 20,
        }
        self.assertEqual(keyframe_count(1.0, **options), 1)
        self.assertEqual(keyframe_count(10.0, **options), 3)
        self.assertEqual(keyframe_count(11.0, **options), 4)
        self.assertEqual(keyframe_count(60.0, **options), 18)
        self.assertEqual(keyframe_count(600.0, **options), 20)

    def test_linear_zero_max_means_unlimited(self) -> None:
        self.assertEqual(keyframe_count(
            600.0,
            strategy="linear",
            keyframes_per_second=0.3,
            min_keyframes_per_scene=1,
            max_keyframes_per_scene=0,
        ), 180)

    def test_selected_frames_are_evenly_spaced_inside_each_scene(self) -> None:
        selected = select_keyframes(
            [(0, 99)],
            fps=10.0,
            strategy="linear",
            keyframes_per_second=0.3,
            min_keyframes_per_scene=1,
            max_keyframes_per_scene=20,
        )
        self.assertEqual(selected, [
            {"frame_number": 17, "scene_index": 0},
            {"frame_number": 50, "scene_index": 0},
            {"frame_number": 83, "scene_index": 0},
        ])

    def test_strategy_change_invalidates_keyframes_and_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video_dir = Path(temporary) / "N001"
            video_dir.mkdir()
            (video_dir / "keyframes.json").write_text(
                json.dumps({"selection": {"strategy": "tiered"}}), encoding="utf-8"
            )
            (video_dir / "embeddings.npy").write_bytes(b"old")

            invalidated = invalidate_mismatched_selection(
                Path(temporary), strategy="linear", keyframes_per_second=0.3,
                min_keyframes_per_scene=1, max_keyframes_per_scene=20,
            )

            self.assertEqual(invalidated, ["N001"])
            self.assertFalse((video_dir / "keyframes.json").exists())
            self.assertFalse((video_dir / "embeddings.npy").exists())

    def test_legacy_keyframes_remain_reusable_for_default_tiered_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video_dir = Path(temporary) / "N001"
            video_dir.mkdir()
            (video_dir / "keyframes.json").write_text("{}", encoding="utf-8")
            (video_dir / "embeddings.npy").write_bytes(b"old")

            invalidated = invalidate_mismatched_selection(
                Path(temporary), strategy="tiered", keyframes_per_second=0.3,
                min_keyframes_per_scene=1, max_keyframes_per_scene=20,
            )

            self.assertEqual(invalidated, [])
            self.assertTrue((video_dir / "embeddings.npy").exists())


if __name__ == "__main__":
    unittest.main()
