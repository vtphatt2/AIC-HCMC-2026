from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocess.pecore.embedding import (
    EmbeddingDataLoaderConfig,
    PECoreEmbeddingConfig,
    PECoreEmbeddingPipeline,
    VisualEmbeddingEncoder,
)
from preprocess.batch.embedding import PECoreEmbeddingStrategy
from preprocess.batch.layout import LotLayout
from preprocess.batch.models import ProcessingResult, VideoAsset


class FakeVisualEncoder(VisualEmbeddingEncoder):
    dimension = 4

    def __init__(self) -> None:
        self.calls: list[tuple[Path, ...]] = []

    def embed(self, image_paths: list[Path]) -> np.ndarray:
        self.calls.append(tuple(image_paths))
        values = []
        for index, _ in enumerate(image_paths, start=1):
            value = np.array([index, 1, 0, 0], dtype=np.float32)
            value /= np.linalg.norm(value)
            values.append(value)
        return np.stack(values).astype(np.float32)


def fake_image_transform(image):
    """Pickle-friendly transform used to exercise the DataLoader path."""
    import torch

    pixel = image.convert("RGB").getpixel((0, 0))
    return torch.tensor([float(pixel[0])], dtype=torch.float32)


class PreparedFakeVisualEncoder(VisualEmbeddingEncoder):
    dimension = 4

    def __init__(self) -> None:
        self.prepared_calls = 0

    def embed(self, image_paths: list[Path]) -> np.ndarray:
        raise AssertionError("the prepared DataLoader path should be used")

    def image_transform(self):
        return fake_image_transform

    def embed_prepared_batch(self, batch, *, non_blocking: bool = False) -> np.ndarray:
        del non_blocking
        self.prepared_calls += 1
        values = np.ones((len(batch), self.dimension), dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class PECoreEmbeddingTests(unittest.TestCase):
    def test_configurable_encoder_writes_sample_compatible_npy_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes" / "L21_V030"
            keyframes.mkdir(parents=True)
            for frame_id in ("000002", "000010", "000100"):
                (keyframes / f"{frame_id}.jpg").write_bytes(b"fake-image")

            first_encoder = FakeVisualEncoder()
            first = PECoreEmbeddingPipeline(first_encoder, batch_size=2).embed_all(
                root / "keyframes",
                root / "PECore-features",
            )
            self.assertEqual(first.image_count, 3)
            self.assertEqual(first.embedded_count, 3)
            self.assertEqual(first.skipped_count, 0)
            self.assertEqual(len(first_encoder.calls), 2)

            output_files = sorted((root / "PECore-features" / "L21_V030").glob("*.npy"))
            self.assertEqual([path.stem for path in output_files], ["000002", "000010", "000100"])
            for path in output_files:
                value = np.load(path, allow_pickle=False)
                self.assertEqual(value.shape, (4,))
                self.assertEqual(value.dtype, np.float32)
                self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0, places=5)

            second_encoder = FakeVisualEncoder()
            second = PECoreEmbeddingPipeline(second_encoder, batch_size=2).embed_all(
                root / "keyframes",
                root / "PECore-features",
            )
            self.assertEqual(second.embedded_count, 0)
            self.assertEqual(second.skipped_count, 3)
            self.assertEqual(second_encoder.calls, [])

    def test_invalid_existing_feature_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes" / "L21_V030"
            keyframes.mkdir(parents=True)
            (keyframes / "000001.jpg").write_bytes(b"fake-image")
            features = root / "PECore-features" / "L21_V030"
            features.mkdir(parents=True)
            np.save(features / "000001.npy", np.zeros(4, dtype=np.float32), allow_pickle=False)

            with self.assertRaises(FileExistsError):
                PECoreEmbeddingPipeline(FakeVisualEncoder()).embed_all(
                    root / "keyframes",
                    root / "PECore-features",
                )

    def test_default_model_contract_matches_sample(self) -> None:
        config = PECoreEmbeddingConfig()
        self.assertEqual(config.model_id, "hf-hub:timm/PE-Core-bigG-14-448")
        self.assertEqual(config.expected_dim, 1280)
        self.assertEqual(config.features_dir_name, "PECore-features")

    def test_dataloader_config_is_nested_and_validated(self) -> None:
        config = PECoreEmbeddingConfig(
            dataloader={
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
            }
        )
        self.assertIsInstance(config.dataloader, EmbeddingDataLoaderConfig)
        self.assertEqual(config.dataloader.num_workers, 2)
        self.assertTrue(config.dataloader.pin_memory)
        with self.assertRaises(ValueError):
            EmbeddingDataLoaderConfig(persistent_workers=True)

    def test_prepared_encoder_uses_configured_dataloader_path(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes" / "L21_V030"
            keyframes.mkdir(parents=True)
            for frame_id, pixel in (("000001", 10), ("000002", 20), ("000003", 30)):
                Image.new("RGB", (2, 2), (pixel, 0, 0)).save(keyframes / f"{frame_id}.png")

            encoder = PreparedFakeVisualEncoder()
            result = PECoreEmbeddingPipeline(
                encoder,
                batch_size=2,
                dataloader=EmbeddingDataLoaderConfig(num_workers=0, pin_memory=False),
            ).embed_all(root / "keyframes", root / "PECore-features")

        self.assertEqual(result.embedded_count, 3)
        self.assertEqual(encoder.prepared_calls, 2)

    def test_batch_strategy_uses_configured_profile_and_feature_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = LotLayout(root / "data", "L29_a")
            layout.create_runtime_dirs()
            keyframe_dir = layout.dataset_dir / "keyframes" / "L21_V030"
            keyframe_dir.mkdir(parents=True)
            (keyframe_dir / "000001.jpg").write_bytes(b"fake-image")
            source = layout.source_root / "L21_V030.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video")
            asset = VideoAsset("L21_V030", source, "L29_a", source.name)
            selection = layout.dataset_dir / "selection-manifests" / "L21_V030.json"
            rendered = keyframe_dir / "manifest.json"
            result = ProcessingResult(asset, {}, 1, selection, rendered)
            config = PECoreEmbeddingConfig(enabled=True, expected_dim=4, batch_size=1)

            batch_result = PECoreEmbeddingStrategy(
                FakeVisualEncoder(),
                config,
                input_profile_id="keyframes",
            ).embed(layout, [asset], [result])

            self.assertEqual(batch_result.embedded_count, 1)
            self.assertTrue(
                (layout.dataset_dir / "PECore-features" / "L21_V030" / "000001.npy").is_file()
            )


if __name__ == "__main__":
    unittest.main()
