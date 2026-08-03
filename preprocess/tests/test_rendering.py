from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from preprocess.batch.config import BatchConfig
from preprocess.keyframes.contracts import (
    ExtractionOutput,
    FrameRef,
    RenderProfile,
    SelectedFrame,
    VideoInfo,
    VideoSource,
)
from preprocess.keyframes.extractors.ffmpeg import FFmpegKeyframeExtractor
from preprocess.keyframes.rendering import render_image
from preprocess.progress import ProgressConfig, TqdmProgressReporter


class RenderingAndProgressTests(unittest.TestCase):
    def test_default_batch_profile_preserves_decoded_dimensions(self) -> None:
        config = BatchConfig.from_json(Path("preprocess/batch/config.example.json"))
        profile = config.processing.render_profile()
        self.assertIsNone(profile.target_short_edge_px)
        self.assertEqual(profile.image_format, "png")
        self.assertEqual(profile.jpeg_quality, 100)
        self.assertFalse(config.progress.leave)
        self.assertEqual(config.embedding.dataloader.prefetch_factor, 2)

    def test_render_profile_preserves_dimensions_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "frame.png"
            with Image.new("RGB", (1920, 1080), (1, 2, 3)) as source:
                size = render_image(source, destination, RenderProfile("original"))
            with Image.open(destination) as rendered:
                self.assertEqual(size, (1920, 1080))
                self.assertEqual(rendered.size, (1920, 1080))
                self.assertEqual(rendered.format, "PNG")

    def test_progress_reporter_can_be_disabled_without_changing_iteration(self) -> None:
        reporter = TqdmProgressReporter(ProgressConfig(enabled=False))
        values = list(reporter.iterate([1, 2, 3], total=3, desc="test", unit="item"))
        self.assertEqual(values, [1, 2, 3])

    def test_ffmpeg_renderer_remaps_frame_indexes_after_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "L21_V001.mp4"
            source_path.write_bytes(b"video")
            source = VideoSource("L21_V001", source_path)
            selected = [
                SelectedFrame(
                    FrameRef("L21_V001", 10, 1_000, 1.0),
                    score=1.0,
                    reasons=("test",),
                    rank=0,
                ),
                SelectedFrame(
                    FrameRef("L21_V001", 20, 2_000, 2.0),
                    score=1.0,
                    reasons=("test",),
                    rank=1,
                ),
            ]
            extractor = FFmpegKeyframeExtractor(
                progress=TqdmProgressReporter(ProgressConfig(enabled=False))
            )
            render_calls: list[tuple[int, ...]] = []

            def fake_render(_source, frame_numbers, destination_dir):
                render_calls.append(tuple(frame_numbers))
                destination_dir.mkdir(parents=True, exist_ok=True)
                count = 1 if len(render_calls) == 1 else len(frame_numbers)
                paths = []
                for index in range(count):
                    path = destination_dir / f"decoded-{index + 1:06d}.png"
                    with Image.new("RGB", (4, 4), (index, 0, 0)) as image:
                        image.save(path, format="PNG")
                    paths.append(path)
                return paths

            output = ExtractionOutput(
                rendered_root=root / "dataset",
                selection_manifest_path=root / "selection.json",
                profile=RenderProfile("original"),
            )
            with patch.object(
                extractor,
                "_render_selected_frame_numbers",
                side_effect=fake_render,
            ), patch.object(
                extractor,
                "_map_selected_to_decoder_indexes",
                return_value=[110, 220],
            ):
                result = extractor.materialize(
                    source,
                    VideoInfo("L21_V001", 3_000, 25.0, 4, 4, 75, "h264"),
                    selected,
                    output,
                )

            self.assertEqual(render_calls, [(10, 20), (110, 220)])
            self.assertEqual(
                [frame.ref.source_frame_number for frame in result.written],
                [10, 20],
            )
            self.assertTrue((root / "dataset" / "original" / "L21_V001" / "000010.png").is_file())
            self.assertTrue((root / "dataset" / "original" / "L21_V001" / "000020.png").is_file())

    def test_ffmpeg_showinfo_timeline_parser_reads_decoder_indexes(self) -> None:
        extractor = FFmpegKeyframeExtractor(
            progress=TqdmProgressReporter(ProgressConfig(enabled=False))
        )
        completed = type(
            "Completed",
            (),
            {
                "stderr": (
                    "[Parsed_showinfo_0 @ 0x1] n:   0 pts:      0 "
                    "pts_time:0 pos:0\n"
                    "[Parsed_showinfo_0 @ 0x1] n:  12 pts:    480 "
                    "pts_time:0.5 pos:1\n"
                )
            },
        )()
        with patch.object(extractor, "_run", return_value=completed):
            timeline = extractor._authoritative_decoder_timeline(
                VideoSource("L21_V001", Path("video.mp4"))
            )
        self.assertEqual(timeline, [(0, 0.0), (12, 0.5)])


if __name__ == "__main__":
    unittest.main()
