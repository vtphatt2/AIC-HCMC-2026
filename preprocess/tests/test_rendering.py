from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from preprocess.batch.config import BatchConfig
from preprocess.keyframes.contracts import RenderProfile
from preprocess.keyframes.rendering import render_image
from preprocess.progress import ProgressConfig, TqdmProgressReporter


class RenderingAndProgressTests(unittest.TestCase):
    def test_default_batch_profile_preserves_decoded_dimensions(self) -> None:
        config = BatchConfig.from_json(Path("preprocess/batch/config.example.json"))
        profile = config.processing.render_profile()
        self.assertIsNone(profile.target_short_edge_px)
        self.assertEqual(profile.image_format, "png")
        self.assertEqual(profile.jpeg_quality, 100)

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


if __name__ == "__main__":
    unittest.main()
