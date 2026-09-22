from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from preprocess.zip_pipeline import (
    ResultArchiveValidator,
    ZipPipelineConfig,
    build_engine_command,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _npy_bytes(shape: tuple[int, ...]) -> bytes:
    header = repr({"descr": "<f4", "fortran_order": False, "shape": shape})
    padding = 16 - ((10 + len(header) + 1) % 16)
    encoded = (header + " " * padding + "\n").encode("latin1")
    count = 1
    for dimension in shape:
        count *= dimension
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded)) + encoded + b"\0" * (count * 4)


class ZipPipelineConfigTest(unittest.TestCase):
    def test_requires_exactly_one_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ZipPipelineConfig()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ZipPipelineConfig(source_url="https://example/Videos_L30_a.zip", source_zip=Path("x.zip"))

    def test_builds_parallel_zip_native_engine_command(self) -> None:
        config = ZipPipelineConfig(
            source_zip=Path("/data/Videos_L30_a.zip"),
            work_root=Path("/work"),
            archive=Path("/results/L30_a_results.zip"),
            limit=2,
        )
        command = build_engine_command(config, repository_root=REPOSITORY_ROOT)
        self.assertEqual(command[0:2], ["bash", str(REPOSITORY_ROOT / "keyframe_pipeline_global_v9_3/run_pipeline.sh")])
        self.assertIn("--zip", command)
        self.assertNotIn("--url", command)
        self.assertIn("--parallel-stages", command)
        self.assertEqual(command[command.index("--limit") + 1], "2")

    def test_forwards_linear_keyframe_configuration(self) -> None:
        config = ZipPipelineConfig(
            source_zip=Path("/data/Video_N001-N010.zip"),
            keyframe_strategy="linear",
            keyframes_per_second=0.4,
            min_keyframes_per_scene=2,
            max_keyframes_per_scene=12,
        )

        command = build_engine_command(config, repository_root=REPOSITORY_ROOT)

        self.assertEqual(command[command.index("--keyframe-strategy") + 1], "linear")
        self.assertEqual(command[command.index("--keyframes-per-second") + 1], "0.4")
        self.assertEqual(command[command.index("--min-keyframes-per-scene") + 1], "2")
        self.assertEqual(command[command.index("--max-keyframes-per-scene") + 1], "12")

    def test_rejects_invalid_linear_keyframe_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "keyframes_per_second"):
            ZipPipelineConfig(source_zip=Path("input.zip"), keyframes_per_second=0)
        with self.assertRaisesRegex(ValueError, "max_keyframes_per_scene"):
            ZipPipelineConfig(
                source_zip=Path("input.zip"),
                min_keyframes_per_scene=3,
                max_keyframes_per_scene=2,
            )

    def test_zero_max_keyframes_is_an_unlimited_valid_configuration(self) -> None:
        config = ZipPipelineConfig(
            source_zip=Path("input.zip"),
            keyframe_strategy="linear",
            max_keyframes_per_scene=0,
        )
        self.assertEqual(config.max_keyframes_per_scene, 0)

    def test_default_archive_uses_challenge_name(self) -> None:
        config = ZipPipelineConfig(
            source_url="https://example.test/path/Videos_L31_b.zip?token=abc",
            work_root=Path("data/zip-preprocess"),
        )
        self.assertEqual(config.result_archive, Path("data/zip-preprocess/L31_b_results.zip"))

    def test_normalizes_new_video_range_archive_name(self) -> None:
        config = ZipPipelineConfig(
            source_zip=Path("/data/Video_N001-N010.zip"),
            work_root=Path("data/zip-preprocess"),
        )
        self.assertEqual(config.data_id, "N001-N010")
        self.assertEqual(config.result_archive, Path("data/zip-preprocess/N001-N010_results.zip"))


class ResultArchiveValidatorTest(unittest.TestCase):
    def test_accepts_challenge_format_without_images_or_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            archive = Path(temp_name) / "L30_a_results.zip"
            manifest = {
                "format_version": 3,
                "num_videos": 1,
                "videos": [{
                    "video_id": "video__L30_V001",
                    "video": "L30_V001.mp4",
                    "num_keyframes": 2,
                    "scenes": "phase1_transnet/video__L30_V001/scenes.json",
                    "keyframes": "phase1_transnet/video__L30_V001/keyframes.json",
                    "embeddings": "phase2_embeddings/video__L30_V001/embeddings.npy",
                }],
            }
            keyframes = {"num_keyframes": 2, "keyframes": [{"frame_number": 1}, {"frame_number": 8}]}
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("manifest.json", json.dumps(manifest))
                output.writestr(manifest["videos"][0]["scenes"], "{}")
                output.writestr(manifest["videos"][0]["keyframes"], json.dumps(keyframes))
                output.writestr(manifest["videos"][0]["embeddings"], _npy_bytes((2, 1280)))

            inspection = ResultArchiveValidator().validate(archive)
            self.assertEqual(inspection.format_version, 3)
            self.assertEqual(inspection.video_count, 1)
            self.assertEqual(inspection.keyframe_count, 2)
            self.assertEqual(inspection.embedding_dim, 1280)

    def test_rejects_media_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            archive = Path(temp_name) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("manifest.json", json.dumps({"format_version": 3, "num_videos": 0, "videos": []}))
                output.writestr("keyframes/frame.jpg", b"jpeg")
            with self.assertRaisesRegex(ValueError, "media artifact"):
                ResultArchiveValidator().validate(archive)


if __name__ == "__main__":
    unittest.main()
