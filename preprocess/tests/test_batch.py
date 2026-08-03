from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from preprocess.batch.archive_extractor import ZipArchiveExtractor
from preprocess.batch.archive_validator import ZipArchiveValidator
from preprocess.batch.checkpoints import CheckpointStore
from preprocess.batch.cleanup import CleanupManager
from preprocess.batch.config import (
    BatchConfig,
    CleanupConfig,
    DownloadConfig,
    LinearSelectionConfig,
    UploadConfig,
)
from preprocess.batch.downloader import Aria2ArchiveDownloader
from preprocess.batch.kaggle_uploader import KaggleStagingStrategy
from preprocess.batch.layout import LotLayout
from preprocess.batch.links import LinkListParser
from preprocess.batch.metadata import JsonMetadataProvider
from preprocess.batch.models import (
    ArchiveInput,
    ArchiveInspection,
    BatchState,
    ProcessingResult,
    UploadResult,
    VideoAsset,
)
from preprocess.batch.shot_boundaries import (
    ShotBoundaryDetection,
    ShotBoundaryDetector,
    ShotBoundaryPipeline,
    load_scene_segments,
)
from preprocess.batch.video_discovery import VideoDiscovery
from preprocess.keyframes.contracts import FrameCandidate, FrameRef, SceneSegment, VideoInfo
from preprocess.keyframes.selectors.linear_rulebase import LinearRuleBasedSelector
from preprocess.progress import ProgressConfig, TqdmProgressReporter


class BatchModuleTests(unittest.TestCase):
    def test_linear_selector_auto_configures_transnet_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping(
                {
                    "data_root": str(root / "data"),
                    "processing": {"selector": "linear-rulebase"},
                },
                base_dir=root,
            )
        self.assertTrue(config.shot_boundary.enabled)
        self.assertEqual(config.processing.scene_segments_dir, root / "data" / "scene-segments")
        self.assertEqual(config.shot_boundary.output_dir, root / "data" / "scene-segments")

    def test_shot_boundary_pipeline_writes_and_reuses_manifest(self) -> None:
        class FakeDetector(ShotBoundaryDetector):
            name = "fake"

            def __init__(self) -> None:
                self.calls = 0

            def detect(self, asset: VideoAsset) -> ShotBoundaryDetection:
                self.calls += 1
                return ShotBoundaryDetection(
                    video_id=asset.video_id,
                    backend=self.name,
                    threshold=0.5,
                    fps=25.0,
                    segments=({"start_ms": 0, "end_ms": 1000},),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "L21_V030.mp4"
            video.write_bytes(b"video")
            asset = VideoAsset("L21_V030", video, "L29_a", video.name)
            detector = FakeDetector()
            pipeline = ShotBoundaryPipeline(
                detector,
                root / "scene-segments",
                progress=TqdmProgressReporter(ProgressConfig(enabled=False)),
            )

            first = pipeline.run([asset])
            second = pipeline.run([asset])
            segments = load_scene_segments(root / "scene-segments" / "L21_V030.json")

        self.assertEqual(detector.calls, 1)
        self.assertFalse(first[0].cached)
        self.assertTrue(second[0].cached)
        self.assertEqual(first[0].scene_count, 1)
        self.assertEqual(segments, [SceneSegment(0, 1000)])

    def test_aria2_command_uses_sixteen_connections_and_splits(self) -> None:
        request = ArchiveInput(
            url="https://aic-data.example/Videos_L29_a.zip",
            archive_name="Videos_L29_a.zip",
            lot_id="L29_a",
            line_number=1,
        )
        command = Aria2ArchiveDownloader(
            config=DownloadConfig(max_concurrent_connections=16, split_count=16)
        ).build_command(request, Path("data/L29_a/archive"))
        self.assertEqual(command[command.index("--split") + 1], "16")
        self.assertEqual(
            command[command.index("--max-connection-per-server") + 1],
            "16",
        )

    def test_links_preserve_lot_id_from_archive_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "links.txt"
            path.write_text(
                "# comment\nhttps://aic-data.example/Videos_L29_a.zip?download=1\n",
                encoding="utf-8",
            )
            request = LinkListParser().parse(path)[0]
        self.assertEqual(request.archive_name, "Videos_L29_a.zip")
        self.assertEqual(request.lot_id, "L29_a")

    def test_zip_validation_and_root_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "Videos_L29_a.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("video/L21_V030.mp4", b"not-a-real-video")
            inspection = ZipArchiveValidator().validate(archive_path)
            extracted = ZipArchiveExtractor().extract(inspection, root / "source", "L29_a")
            self.assertEqual(extracted.name, "L29_a")
            self.assertTrue((extracted / "L21_V030.mp4").is_file())

    def test_zip_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("video/../escape.mp4", b"bad")
            with self.assertRaises(ValueError):
                ZipArchiveValidator().validate(archive_path)

    def test_video_discovery_uses_original_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "L29_a"
            source.mkdir()
            video = source / "L21_V030.mp4"
            video.write_bytes(b"video")
            asset = VideoDiscovery([".mp4"]).discover(source, "L29_a")[0]
        self.assertEqual(asset.video_id, "L21_V030")
        self.assertEqual(asset.source_name, "L21_V030.mp4")

    def test_linear_rulebase_uses_configured_duration_buckets(self) -> None:
        rule = LinearSelectionConfig(
            short_duration_ms=1_000,
            short_frame_count=1,
            base_duration_ms=3_000,
            base_frame_count=2,
            increment_duration_ms=3_000,
            increment_frame_count=1,
        )
        self.assertEqual(rule.frame_count(1_000), 1)
        self.assertEqual(rule.frame_count(3_000), 2)
        self.assertEqual(rule.frame_count(6_000), 3)
        self.assertEqual(rule.frame_count(9_000), 4)

        segments = (
            SceneSegment(0, 1_000),
            SceneSegment(1_000, 4_000),
            SceneSegment(4_000, 10_000),
        )
        candidates = [
            FrameCandidate(
                FrameRef("L21_V030", frame_number, frame_number * 100, frame_number * 0.1)
            )
            for frame_number in range(100)
        ]
        selected = LinearRuleBasedSelector(
            segments,
            frame_count_for=rule.frame_count,
            rule_config={"increment_duration_ms": rule.increment_duration_ms},
        ).select(
            VideoInfo("L21_V030", 10_000, 10.0, 320, 240, 100, "mpeg4"),
            candidates,
        )
        counts = [
            sum(item.metadata["scene_index"] == index for item in selected)
            for index in range(3)
        ]
        self.assertEqual(counts, [1, 2, 3])

    def test_checkpoint_transition_is_atomic_and_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary) / "state.json")
            store.transition(BatchState.DOWNLOADING)
            state = store.transition(BatchState.DOWNLOADED, payload={"size": 42})
            loaded = store.load()
        self.assertEqual(state["state"], BatchState.DOWNLOADED.value)
        self.assertEqual(loaded["size"], 42)
        self.assertEqual(len(loaded["events"]), 2)

    def test_stage_checkpoint_resumes_only_the_interrupted_stage(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping({"data_root": str(root / "data")}, base_dir=root)
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            orchestrator.progress = TqdmProgressReporter(ProgressConfig(enabled=False))
            orchestrator.shot_boundary_detector = None
            orchestrator.embedding = None
            request = ArchiveInput(
                url="https://example.test/Videos_L29_a.zip",
                archive_name="Videos_L29_a.zip",
                lot_id="L29_a",
                line_number=1,
            )
            layout = LotLayout(config.data_root, request.lot_id)
            layout.create_runtime_dirs()
            checkpoints = CheckpointStore(layout.state_path)
            orchestrator._initialize_state(checkpoints, request)

            calls: list[str] = []
            orchestrator._execute_stage(
                checkpoints,
                request,
                "download",
                action=lambda: calls.append("download") or "downloaded",
                restore=lambda: calls.append("restore-download") or "downloaded",
            )
            with self.assertRaises(RuntimeError):
                orchestrator._execute_stage(
                    checkpoints,
                    request,
                    "process_validate",
                    action=lambda: (_ for _ in ()).throw(RuntimeError("interrupted")),
                    restore=lambda: (_ for _ in ()).throw(FileNotFoundError("not complete")),
                )

            resumed = orchestrator._execute_stage(
                checkpoints,
                request,
                "process_validate",
                action=lambda: calls.append("process-retry") or "processed",
                restore=lambda: (_ for _ in ()).throw(FileNotFoundError("not complete")),
            )
            orchestrator._execute_stage(
                checkpoints,
                request,
                "download",
                action=lambda: calls.append("download-again") or "wrong",
                restore=lambda: calls.append("restore-download") or "downloaded",
            )
            state = checkpoints.load()

        self.assertEqual(resumed, "processed")
        self.assertEqual(calls, ["download", "process-retry", "restore-download"])
        self.assertEqual(state["stages"]["download"]["status"], "completed")
        self.assertEqual(state["stages"]["process_validate"]["status"], "completed")
        self.assertEqual(state["stages"]["process_validate"]["attempt"], 2)

    def test_staging_excludes_source_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_root = root / "metadata"
            metadata_root.mkdir()
            (metadata_root / "L21_V030.json").write_text('{"video_link": "x"}\n', encoding="utf-8")
            layout = LotLayout(root / "data", "L29_a")
            layout.create_runtime_dirs()
            keyframes = layout.dataset_dir / "keyframes" / "L21_V030"
            keyframes.mkdir(parents=True)
            (keyframes / "000000.jpg").write_bytes(b"jpg")
            (keyframes / "manifest.json").write_text("{}", encoding="utf-8")
            selection = layout.dataset_dir / "selection-manifests" / "L21_V030.json"
            selection.parent.mkdir(parents=True)
            selection.write_text("{}", encoding="utf-8")
            validation = layout.reports_dir / "validation" / "L21_V030.json"
            validation.parent.mkdir(parents=True)
            validation.write_text("{}", encoding="utf-8")
            source_video = layout.source_root / "L21_V030.mp4"
            source_video.parent.mkdir(parents=True)
            source_video.write_bytes(b"raw-video")
            template = root / "dataset-metadata.json"
            template.write_text('{"title":"test"}\n', encoding="utf-8")

            asset = VideoAsset("L21_V030", source_video, "L29_a", source_video.name)
            result = ProcessingResult(asset, {"fps": None}, 1, selection, keyframes / "manifest.json")
            config = UploadConfig(enabled=True, dataset_ref="owner/test", metadata_template=template)
            staging = KaggleStagingStrategy(JsonMetadataProvider(metadata_root), config).stage(
                layout, [asset], [result]
            )
            staged_paths = {path.relative_to(staging.staging_dir).as_posix() for path in staging.files}
        self.assertIn("dataset-metadata.json", staged_paths)
        self.assertIn("keyframes/L21_V030/000000.jpg", staged_paths)
        self.assertIn("manifests/rendered/L21_V030.json", staged_paths)
        self.assertNotIn("videos/L21_V030.mp4", staged_paths)
        self.assertNotIn("source/L29_a/L21_V030.mp4", staged_paths)

    def test_cleanup_requires_verified_upload_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = LotLayout(root / "data", "L29_a")
            layout.create_runtime_dirs()
            for path in (
                layout.archive_dir / "archive.zip",
                layout.source_dir / "L29_a" / "video.mp4",
                layout.dataset_dir / "keyframes" / "L21_V030" / "000000.jpg",
                layout.dataset_dir / "PECore-features" / "L21_V030" / "000000.npy",
                layout.dataset_dir / "selection-manifests" / "L21_V030.json",
                layout.staging_dir / "dataset-metadata.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            metadata = root / "metadata" / "L21_V030.json"
            metadata.parent.mkdir()
            metadata.write_text("{}", encoding="utf-8")
            upload = UploadResult("owner/test", "version", True, ("kaggle",), "ok")
            CleanupManager(CleanupConfig(enabled=True)).cleanup(layout, upload)
            self.assertFalse(layout.archive_dir.exists())
            self.assertFalse(layout.source_dir.exists())
            self.assertFalse((layout.dataset_dir / "selection-manifests").exists())
            self.assertTrue(metadata.is_file())


if __name__ == "__main__":
    unittest.main()
