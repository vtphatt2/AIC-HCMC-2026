from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
import zipfile
import numpy as np
import threading
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from preprocess.batch.archive_extractor import ZipArchiveExtractor
from preprocess.batch.archive_validator import ZipArchiveValidator
from preprocess.batch.checkpoints import CheckpointStore
from preprocess.batch.cleanup import CleanupManager
from preprocess.batch.config import (
    ArchiveConfig,
    BatchConfig,
    CleanupConfig,
    DownloadConfig,
    LinearSelectionConfig,
    UploadConfig,
)
from preprocess.batch.dataset_state import DatasetUploadStateStore
from preprocess.batch.downloader import Aria2ArchiveDownloader
from preprocess.batch.kaggle_uploader import (
    CumulativeKaggleStagingStrategy,
    KaggleCliUploader,
    KaggleStagingStrategy,
)
from preprocess.batch.layout import LotLayout
from preprocess.batch.links import LinkListParser
from preprocess.batch.locks import ExclusiveFileLock
from preprocess.batch.metadata import JsonMetadataProvider
from preprocess.batch.models import (
    ArchiveInput,
    BatchState,
    ProcessingResult,
    StagingResult,
    UploadResult,
    VideoAsset,
)
from preprocess.pecore.embedding import EmbeddingBatchResult, EmbeddingVideoResult, NpyFeatureWriter
from preprocess.batch.provenance import (
    FileDigestCache,
    atomic_json_write,
    digest_directory,
    file_fingerprint_matches,
    sha256_file,
)
from preprocess.batch.shot_boundaries import (
    ShotBoundaryDetection,
    ShotBoundaryDetector,
    ShotBoundaryPipeline,
    load_scene_segments,
)
from preprocess.batch.video_discovery import VideoDiscovery
from preprocess.keyframes.contracts import (
    FrameCandidate,
    FrameRef,
    SceneSegment,
    SelectedFrame,
    VideoInfo,
    VideoSource,
)
from preprocess.keyframes.extractors.ffmpeg import FFmpegKeyframeExtractor
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

    def test_transnet_window_batching_preserves_prediction_order(self) -> None:
        import torch
        from preprocess.batch.shot_boundaries import TransNetV2ShotBoundaryDetector

        frame_count = 123
        raw = np.zeros((frame_count, 27, 48, 3), dtype=np.uint8)
        raw[:, 0, 0, 0] = np.arange(frame_count, dtype=np.uint8)

        class FakeInput:
            def output(self, *args, **kwargs):
                del args, kwargs
                return self

            def run(self, **kwargs):
                del kwargs
                return raw.tobytes(), b""

        fake_ffmpeg = type(
            "FakeFFmpeg",
            (),
            {"input": staticmethod(lambda path: FakeInput())},
        )

        class FakeModel:
            device = "cpu"

            @staticmethod
            def predict_raw(batch):
                values = batch[:, :, 0, 0, 0].float().unsqueeze(-1) / 255.0
                return values, values

        with patch.dict(sys.modules, {"ffmpeg": fake_ffmpeg}):
            _, single, many = TransNetV2ShotBoundaryDetector._predict_video_batched(
                FakeModel(), Path("video.mp4"), 4
            )

        self.assertEqual(tuple(single.shape), (frame_count,))
        self.assertTrue(torch.equal(single, many))
        self.assertTrue(
            torch.equal(single, torch.arange(frame_count, dtype=torch.float32) / 255.0)
        )

    def test_crc_corrupt_zip_is_never_published_by_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "Videos_L21_a.zip"
            payload = b"unique-video-payload"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
                handle.writestr("video/L21_V001.mp4", payload)
            data = bytearray(archive.read_bytes())
            offset = data.index(payload)
            data[offset] ^= 0x01
            archive.write_bytes(data)

            inspection = ZipArchiveValidator(
                ArchiveConfig(video_extensions=(".mp4",))
            ).validate(archive)
            destination = root / "source"
            with self.assertRaises(zipfile.BadZipFile):
                ZipArchiveExtractor().extract(inspection, destination, "L21_a")

            self.assertFalse((destination / "L21_a").exists())

    def test_upload_defaults_to_lot_scoped_auto_dataset_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping(
                {
                    "data_root": str(root / "data"),
                    "upload": {"enabled": True, "dataset_ref": "owner/test"},
                },
                base_dir=root,
            )

        self.assertEqual(config.upload.staging_scope, "lot")
        self.assertEqual(config.upload.mode, "auto")
        self.assertIsNone(config.upload.dataset_staging_dir)

    def test_dataset_ref_template_resolves_one_dataset_per_lot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping(
                {
                    "data_root": str(root / "data"),
                    "upload": {
                        "enabled": True,
                        "mode": "auto",
                        "staging_scope": "lot",
                        "dataset_ref_template": "tdat835/aic2026-hcmc-{lot_slug}",
                    },
                },
                base_dir=root,
            )

        self.assertEqual(
            config.upload.target_for_lot("L22_a").dataset_ref,
            "tdat835/aic2026-hcmc-l22-a",
        )
        self.assertEqual(config.upload.target_for_lot("L22_a").mode, "auto")

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

    def test_zip_duplicate_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "duplicate.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("video/L21_V030.mp4", b"first")
                archive.writestr("video/L21_V030.mp4", b"second")
            with self.assertRaisesRegex(ValueError, "Duplicate ZIP member"):
                ZipArchiveValidator().validate(archive_path)

    def test_zip_member_size_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "large-member.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("video/L21_V030.mp4", b"0123456789")
            validator = ZipArchiveValidator(
                ArchiveConfig(max_member_uncompressed_bytes=5)
            )
            with self.assertRaisesRegex(ValueError, "member exceeds size limit"):
                validator.validate(archive_path)

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

    def test_ffmpeg_remap_prefers_decoder_ordinal_when_pts_ticks_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = VideoSource("L21_V001", root / "L21_V001.mp4")
            source.path.write_bytes(b"video")
            selected = [
                SelectedFrame(
                    FrameRef("L21_V001", 3084, 101_466, 101.466667),
                    score=1.0,
                    reasons=(),
                    rank=0,
                )
            ]
            extractor = FFmpegKeyframeExtractor(
                progress=TqdmProgressReporter(ProgressConfig(enabled=False))
            )
            with patch.object(
                extractor,
                "_authoritative_decoder_timeline",
                return_value=[(3084, 101.467000)],
            ):
                mapped = extractor._map_selected_to_decoder_indexes(source, selected)

        self.assertEqual(mapped, [3084])

    def test_checkpoint_transition_is_atomic_and_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary) / "state.json")
            store.transition(BatchState.DOWNLOADING)
            state = store.transition(BatchState.DOWNLOADED, payload={"size": 42})
            loaded = store.load()
        self.assertEqual(state["state"], BatchState.DOWNLOADED.value)
        self.assertEqual(loaded["size"], 42)
        self.assertEqual(len(loaded["events"]), 2)

    def test_upload_state_is_separate_and_migrates_legacy_upload_stages(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping(
                {"data_root": str(root / "data")},
                base_dir=root,
            )
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            request = ArchiveInput(
                url="https://example.test/Videos_L29_a.zip",
                archive_name="Videos_L29_a.zip",
                lot_id="L29_a",
                line_number=1,
            )
            layout = LotLayout(config.data_root, request.lot_id)
            layout.create_runtime_dirs()
            store = CheckpointStore(layout.upload_state_path)
            orchestrator._initialize_upload_state(
                store,
                request,
                legacy_state={
                    "stages": {
                        "stage_upload": {"status": "completed", "fingerprint": "upload-fp"},
                        "cleanup": {"status": "completed", "fingerprint": "cleanup-fp"},
                    }
                },
            )
            state = store.load()

        self.assertNotEqual(layout.upload_state_path, layout.state_path)
        self.assertTrue(state["migrated_from_state"])
        self.assertEqual(
            state["stages"]["stage_upload"]["fingerprint"],
            "upload-fp",
        )
        self.assertEqual(state["stages"]["cleanup"]["fingerprint"], "cleanup-fp")

    def test_upload_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "upload.lock"
            with ExclusiveFileLock(lock_path, purpose="test"):
                with self.assertRaises(RuntimeError):
                    with ExclusiveFileLock(lock_path, purpose="test"):
                        pass

    def test_upload_only_marks_main_checkpoint_completed(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        class FakeUploadOrchestrator(BatchOrchestrator):
            def _restore_assets(self, layout, request):
                return []

            def _restore_processed(self, assets, layout):
                return []

            def _run_upload_stages(
                self,
                request,
                assets,
                processed,
                layout,
                checkpoints,
                *,
                reuse_completed=False,
            ):
                del request, assets, processed, layout, reuse_completed
                return UploadResult(
                    dataset_ref="owner/test",
                    mode="version",
                    verified=True,
                    command=("kaggle",),
                    output_tail="ok",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig(
                data_root=root / "data",
                upload=UploadConfig(enabled=True, dataset_ref="owner/test"),
            )
            orchestrator = object.__new__(FakeUploadOrchestrator)
            orchestrator.config = config
            orchestrator.embedding = None
            orchestrator.progress = TqdmProgressReporter(ProgressConfig(enabled=False))
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
            checkpoints.start_stage("process_validate", fingerprint="process-fp")
            checkpoints.complete_stage("process_validate", fingerprint="process-fp")

            result = orchestrator.upload_lot(request.lot_id)
            main_state = checkpoints.load()
            upload_state = CheckpointStore(layout.upload_state_path).load()

        self.assertTrue(result.verified)
        self.assertEqual(main_state["state"], BatchState.COMPLETED.value)
        self.assertTrue(main_state["upload_only"])
        self.assertTrue(upload_state["initialized"])

    def test_explicit_upload_command_overrides_disabled_upload_flag(self) -> None:
        from preprocess.batch import cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data_root": str(root / "data"),
                        "upload": {
                            "enabled": False,
                            "dataset_ref": "owner/test",
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake_orchestrator = Mock()
            fake_orchestrator.upload_lot.return_value = UploadResult(
                dataset_ref="owner/test",
                mode="version",
                verified=True,
                command=("kaggle",),
                output_tail="ok",
            )
            with patch(
                "preprocess.batch.cli.build_default_orchestrator",
                return_value=fake_orchestrator,
            ) as builder:
                result = cli._upload_lot(
                    Namespace(config=config_path, lot_id="L29_a")
                )
            effective_config = builder.call_args.args[0]

        self.assertEqual(result, 0)
        self.assertTrue(effective_config.upload.enabled)
        fake_orchestrator.upload_lot.assert_called_once_with("L29_a")

    def test_completed_upload_state_is_reused_without_local_artifacts(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        class FakeUploadOrchestrator(BatchOrchestrator):
            def _restore_assets(self, layout, request):
                self.restore_assets_calls += 1
                if self.fail_restore:
                    raise AssertionError("completed upload should not restore source artifacts")
                return []

            def _restore_processed(self, assets, layout):
                return []

            def _run_upload_stages(
                self,
                request,
                assets,
                processed,
                layout,
                checkpoints,
                *,
                reuse_completed=False,
            ):
                del request, assets, processed, reuse_completed
                upload = UploadResult(
                    dataset_ref="owner/test",
                    mode="version",
                    verified=True,
                    command=("kaggle",),
                    output_tail="ok",
                )
                for name in ("stage_upload", "cleanup"):
                    checkpoints.start_stage(name, fingerprint="stored")
                    checkpoints.complete_stage(name, fingerprint="stored")
                layout.receipts_dir.mkdir(parents=True, exist_ok=True)
                (layout.receipts_dir / "upload.json").write_text(
                    json.dumps(upload.to_dict()),
                    encoding="utf-8",
                )
                (layout.receipts_dir / "cleanup.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                checkpoints.transition(BatchState.COMPLETED)
                return upload

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig(
                data_root=root / "data",
                upload=UploadConfig(enabled=True, dataset_ref="owner/test"),
            )
            orchestrator = object.__new__(FakeUploadOrchestrator)
            orchestrator.config = config
            orchestrator.embedding = None
            orchestrator.progress = TqdmProgressReporter(ProgressConfig(enabled=False))
            orchestrator.restore_assets_calls = 0
            orchestrator.fail_restore = False
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
            checkpoints.start_stage("process_validate", fingerprint="process-fp")
            checkpoints.complete_stage("process_validate", fingerprint="process-fp")

            first_result = orchestrator.upload_lot(request.lot_id)
            orchestrator.fail_restore = True
            second_result = orchestrator.upload_lot(request.lot_id)

        self.assertTrue(first_result.verified)
        self.assertEqual(first_result, second_result)
        self.assertEqual(orchestrator.restore_assets_calls, 1)

    def test_run_all_skips_completed_lots_and_runs_pending_lots(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        class RecordingProgress:
            def __init__(self) -> None:
                self.total_units = None
                self.skipped: list[tuple[str, int]] = []

            def start_pipeline(self, *, total_units, total_lots, stage_names) -> None:
                del total_lots, stage_names
                self.total_units = total_units

            def set_lot_context(self, *, lot_index, total_lots, lot_id) -> None:
                del lot_index, total_lots, lot_id

            def skip_lot(self, *, lot_id, stage_count) -> None:
                self.skipped.append((lot_id, stage_count))

            def finish_pipeline(self) -> None:
                return None

        class FakeOrchestrator(BatchOrchestrator):
            def run_lot(self, request):
                self.run_calls.append(request.lot_id)
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig(data_root=root / "data")
            orchestrator = object.__new__(FakeOrchestrator)
            orchestrator.config = config
            orchestrator.shot_boundary_detector = None
            orchestrator.embedding = None
            orchestrator.progress = RecordingProgress()
            orchestrator.run_calls = []
            completed = ArchiveInput(
                url="https://example.test/Videos_L29_a.zip",
                archive_name="Videos_L29_a.zip",
                lot_id="L29_a",
                line_number=1,
            )
            pending = ArchiveInput(
                url="https://example.test/Videos_L30_a.zip",
                archive_name="Videos_L30_a.zip",
                lot_id="L30_a",
                line_number=2,
            )
            completed_layout = LotLayout(config.data_root, completed.lot_id)
            completed_layout.create_runtime_dirs()
            CheckpointStore(completed_layout.state_path).write(
                {
                    "state": BatchState.COMPLETED.value,
                    "initialized": True,
                    "request": completed.to_dict(),
                    "events": [],
                }
            )

            with patch("preprocess.batch.orchestrator.PreflightChecker.run"):
                results = orchestrator.run_all([completed, pending])

        self.assertEqual(results, [None])
        self.assertEqual(orchestrator.run_calls, ["L30_a"])
        self.assertEqual(orchestrator.progress.skipped, [("L29_a", 5)])
        self.assertEqual(orchestrator.progress.total_units, 10)

    def test_overlap_upload_is_bounded_and_runs_during_next_lot(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator, LotRunResult

        started = threading.Event()
        allow_first_upload = threading.Event()

        class FakeOrchestrator(BatchOrchestrator):
            def _pipeline_stage_names(self):
                return ("process", "stage_upload", "cleanup")

            def _is_completed_lot(self, request):
                del request
                return False

            def run_lot(self, request, *, defer_upload=False):
                self.deferred.append(defer_upload)
                if request.lot_id == "L22_a":
                    self.overlapped = started.wait(timeout=2)
                    allow_first_upload.set()
                return LotRunResult(request.lot_id, (), (), None, None)

            def _upload_lot_in_background(self, lot_id):
                if lot_id == "L21_a":
                    started.set()
                    if not allow_first_upload.wait(timeout=2):
                        raise RuntimeError("next lot did not overlap first upload")
                return UploadResult(lot_id, "create", True, ("kaggle",), "ok")

        config = BatchConfig.from_mapping(
            {
                "upload": {
                    "enabled": True,
                    "dataset_ref_template": "owner/test-{lot_slug}",
                },
                "scheduling": {"overlap_upload": True},
            }
        )
        orchestrator = object.__new__(FakeOrchestrator)
        orchestrator.config = config
        orchestrator.progress = Mock()
        orchestrator.deferred = []
        orchestrator.overlapped = False
        requests = [
            ArchiveInput(
                f"https://example.test/Videos_{lot_id}.zip",
                f"Videos_{lot_id}.zip",
                lot_id,
                index,
            )
            for index, lot_id in enumerate(("L21_a", "L22_a"), start=1)
        ]
        with patch("preprocess.batch.orchestrator.PreflightChecker.run"):
            results = orchestrator.run_all(requests)

        self.assertEqual(orchestrator.deferred, [True, True])
        self.assertTrue(orchestrator.overlapped)
        self.assertEqual(
            [result.upload.dataset_ref for result in results if result.upload],
            ["L21_a", "L22_a"],
        )

    def test_render_embedding_overlap_is_bounded_and_checkpointed(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        embedding_started = threading.Event()
        allow_embedding = threading.Event()

        class FakeProcessor:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.rendered_second_during_embedding = False

            def process(self, assets, layout):
                asset = assets[0]
                self.calls.append(asset.video_id)
                if asset.video_id == "L21_V002":
                    self.rendered_second_during_embedding = embedding_started.wait(
                        timeout=2
                    )
                    allow_embedding.set()
                keyframes = layout.dataset_dir / "keyframes" / asset.video_id
                keyframes.mkdir(parents=True, exist_ok=True)
                (keyframes / "000001.jpg").write_bytes(asset.video_id.encode())
                rendered = keyframes / "manifest.json"
                rendered.write_text("{}", encoding="utf-8")
                selection = (
                    layout.dataset_dir
                    / "selection-manifests"
                    / f"{asset.video_id}.json"
                )
                selection.parent.mkdir(parents=True, exist_ok=True)
                selection.write_text("{}", encoding="utf-8")
                return [
                    ProcessingResult(
                        asset,
                        {"duration_ms": 1},
                        1,
                        selection,
                        rendered,
                    )
                ]

        class PassingValidator:
            @staticmethod
            def validate(_context):
                return type(
                    "Report",
                    (),
                    {"passed": True, "to_dict": lambda self: {"passed": True}},
                )()

        class FakeEmbedding:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.pipeline = type(
                    "Pipeline", (), {"cache_fingerprint": "fake-cache"}
                )()
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def for_background(self):
                return self

            def embed(self, layout, assets, results):
                del results
                asset = assets[0]
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    self.calls.append(asset.video_id)
                    if asset.video_id == "L21_V001":
                        embedding_started.set()
                        if not allow_embedding.wait(timeout=2):
                            raise RuntimeError("second render did not overlap embedding")
                    source_dir = layout.dataset_dir / "keyframes" / asset.video_id
                    output_dir = (
                        layout.dataset_dir / "PECore-features" / asset.video_id
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    feature = output_dir / "000001.npy"
                    np.save(feature, np.array([1, 0, 0, 0], dtype=np.float32))
                    NpyFeatureWriter(4).write_video_metadata(
                        output_dir,
                        {
                            "000001": {
                                "schema_version": 1,
                                "image_sha256": sha256_file(
                                    source_dir / "000001.jpg"
                                ),
                                "cache_fingerprint": "fake-cache",
                            }
                        },
                    )
                    video = EmbeddingVideoResult(
                        asset.video_id,
                        source_dir,
                        output_dir,
                        1,
                        1,
                        0,
                        4,
                        (feature,),
                        "fake-cache",
                    )
                    return EmbeddingBatchResult(
                        (video,), 4, "fake-cache", {"worker": "fake"}
                    )
                finally:
                    with self.lock:
                        self.active -= 1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping(
                {
                    "data_root": str(root / "data"),
                    "embedding": {"enabled": True, "expected_dim": 4},
                    "scheduling": {"overlap_render_embedding": True},
                },
                base_dir=root,
            )
            layout = LotLayout(config.data_root, "L21_a")
            layout.create_runtime_dirs()
            assets = []
            for video_id in ("L21_V001", "L21_V002"):
                source = layout.source_root / f"{video_id}.mp4"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(video_id.encode())
                assets.append(VideoAsset(video_id, source, "L21_a", source.name))
            request = ArchiveInput(
                "https://example.test/Videos_L21_a.zip",
                "Videos_L21_a.zip",
                "L21_a",
                1,
            )
            checkpoints = CheckpointStore(layout.state_path)
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            orchestrator.embedding = FakeEmbedding()
            orchestrator.processor = FakeProcessor()
            orchestrator.video_validator = PassingValidator()
            orchestrator.metadata_provider = object()
            orchestrator.shot_boundary_detector = None
            orchestrator.progress = TqdmProgressReporter(
                ProgressConfig(enabled=False)
            )
            orchestrator._cached_runtime_signature = {"test": "runtime"}
            orchestrator._initialize_state(checkpoints, request)

            with patch(
                "preprocess.batch.orchestrator.load_video_metadata",
                return_value={},
            ):
                processed, embedding = orchestrator._process_and_embed_overlapped(
                    request,
                    assets,
                    layout,
                    checkpoints,
                )
            state = checkpoints.load()
            journal_exists = (
                layout.reports_dir
                / "embedding-completions"
                / "L21_V001.json"
            ).is_file()

        self.assertTrue(orchestrator.processor.rendered_second_during_embedding)
        self.assertEqual(orchestrator.embedding.max_active, 1)
        self.assertEqual(orchestrator.embedding.calls, ["L21_V001", "L21_V002"])
        self.assertEqual(
            [item.asset.video_id for item in processed],
            ["L21_V001", "L21_V002"],
        )
        self.assertEqual(embedding.image_count, 2)
        self.assertEqual(state["stages"]["process_validate"]["status"], "completed")
        self.assertEqual(state["stages"]["embedding"]["status"], "completed")
        self.assertTrue(journal_exists)

    def test_run_all_does_not_skip_completed_lot_for_changed_request(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig(data_root=root / "data")
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            request = ArchiveInput(
                url="https://example.test/Videos_L29_a.zip",
                archive_name="Videos_L29_a.zip",
                lot_id="L29_a",
                line_number=1,
            )
            changed_request = ArchiveInput(
                url="https://example.test/Videos_L29_a-replacement.zip",
                archive_name="Videos_L29_a-replacement.zip",
                lot_id="L29_a",
                line_number=1,
            )
            layout = LotLayout(config.data_root, request.lot_id)
            layout.create_runtime_dirs()
            CheckpointStore(layout.state_path).write(
                {
                    "state": BatchState.COMPLETED.value,
                    "initialized": True,
                    "request": request.to_dict(),
                    "events": [],
                }
            )

            with self.assertRaisesRegex(RuntimeError, "different archive request"):
                orchestrator._is_completed_lot(changed_request)

    def test_completed_local_lot_is_reopened_when_stage_code_fingerprint_changes(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        class FingerprintOrchestrator(BatchOrchestrator):
            def _pipeline_stage_names(self):
                return ("download",)

            def _stage_fingerprint(self, request, name):
                del request, name
                return "new-code"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig(data_root=root / "data")
            request = ArchiveInput(
                "https://example.test/Videos_L21_a.zip",
                "Videos_L21_a.zip",
                "L21_a",
                1,
            )
            layout = LotLayout(config.data_root, request.lot_id)
            layout.create_runtime_dirs()
            orchestrator = object.__new__(FingerprintOrchestrator)
            orchestrator.config = config
            CheckpointStore(layout.state_path).write(
                {
                    "state": BatchState.COMPLETED.value,
                    "request": request.to_dict(),
                    "config_fingerprint": orchestrator._config_fingerprint(),
                    "stages": {
                        "download": {
                            "status": "completed",
                            "fingerprint": "old-code",
                        }
                    },
                }
            )
            completed = orchestrator._is_completed_lot(request)

        self.assertFalse(completed)

    def test_completed_stage_records_elapsed_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary) / "state.json")
            store.start_stage("embedding", fingerprint="fingerprint")
            state = store.complete_stage("embedding", fingerprint="fingerprint")

        stage = state["stages"]["embedding"]
        self.assertIn("elapsed_seconds", stage)
        self.assertGreaterEqual(stage["elapsed_seconds"], 0.0)
        completed_event = state["events"][-1]
        self.assertEqual(completed_event["status"], "completed")
        self.assertIn("elapsed_seconds", completed_event["payload"])

    def test_video_checkpoint_survives_retry_of_same_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary) / "state.json")
            store.start_stage("process_validate", fingerprint="stage-fingerprint")
            store.start_video("process_validate", "L21_V001", fingerprint="video-fingerprint")
            store.complete_video(
                "process_validate",
                "L21_V001",
                fingerprint="video-fingerprint",
                payload={"asset": {"video_id": "L21_V001"}},
            )
            store.start_stage("process_validate", fingerprint="stage-fingerprint")
            state = store.load()
            video_is_complete = store.video_is_complete(
                "process_validate",
                "L21_V001",
                "video-fingerprint",
            )

        self.assertTrue(video_is_complete)
        self.assertEqual(state["stages"]["process_validate"]["attempt"], 2)
        self.assertEqual(
            state["stages"]["process_validate"]["videos"]["L21_V001"]["attempt"],
            1,
        )

    def test_process_validate_resumes_after_completed_video(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        class FakeProcessor:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.fail_video_id: str | None = "L21_V002"

            def process(self, assets, layout):
                asset = assets[0]
                self.calls.append(asset.video_id)
                if asset.video_id == self.fail_video_id:
                    raise RuntimeError("interrupted after first video")
                selection = layout.dataset_dir / "selection-manifests" / f"{asset.video_id}.json"
                rendered = layout.dataset_dir / "keyframes" / asset.video_id / "manifest.json"
                selection.parent.mkdir(parents=True, exist_ok=True)
                rendered.parent.mkdir(parents=True, exist_ok=True)
                stat = asset.path.stat()
                selection.write_text(
                    json.dumps(
                        {
                            "source": {
                                "fingerprint": {
                                    "size_bytes": stat.st_size,
                                    "mtime_ns": stat.st_mtime_ns,
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                rendered.write_text(json.dumps({"frames": []}), encoding="utf-8")
                return [ProcessingResult(asset, {"duration_ms": 1}, 0, selection, rendered)]

        class PassingValidator:
            @staticmethod
            def validate(_context):
                return type("Report", (), {"passed": True, "to_dict": lambda self: {"passed": True}})()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping({"data_root": str(root / "data")}, base_dir=root)
            layout = LotLayout(config.data_root, "L29_a")
            layout.create_runtime_dirs()
            assets = []
            for video_id in ("L21_V001", "L21_V002"):
                path = layout.source_root / f"{video_id}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(video_id.encode("utf-8"))
                assets.append(VideoAsset(video_id, path, "L29_a", path.name))
            request = ArchiveInput(
                url="https://example.test/Videos_L29_a.zip",
                archive_name="Videos_L29_a.zip",
                lot_id="L29_a",
                line_number=1,
            )
            checkpoints = CheckpointStore(layout.state_path)
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            orchestrator.shot_boundary_detector = None
            orchestrator.progress = TqdmProgressReporter(ProgressConfig(enabled=False))
            orchestrator.metadata_provider = object()
            orchestrator.processor = FakeProcessor()
            orchestrator.video_validator = PassingValidator()
            orchestrator._initialize_state(checkpoints, request)
            stage_fingerprint = orchestrator._stage_fingerprint(request, "process_validate")
            checkpoints.start_stage("process_validate", fingerprint=stage_fingerprint)

            with patch("preprocess.batch.orchestrator.load_video_metadata", return_value={}):
                with self.assertRaises(RuntimeError):
                    orchestrator._process_and_validate(request, assets, layout, checkpoints)

                orchestrator.processor.fail_video_id = None
                checkpoints.start_stage("process_validate", fingerprint=stage_fingerprint)
                orchestrator._process_and_validate(request, assets, layout, checkpoints)

            state = checkpoints.load()

        self.assertEqual(orchestrator.processor.calls, ["L21_V001", "L21_V002", "L21_V002"])
        self.assertEqual(
            state["stages"]["process_validate"]["videos"]["L21_V001"]["status"],
            "completed",
        )
        self.assertEqual(
            state["stages"]["process_validate"]["videos"]["L21_V002"]["status"],
            "completed",
        )

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

    def test_stage_fingerprint_scopes_shared_inputs_to_current_lot(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            metadata_root = root / "metadata"
            scene_root = root / "scene-segments"
            metadata_root.mkdir()
            scene_root.mkdir()
            config = BatchConfig.from_mapping(
                {
                    "data_root": str(data_root),
                    "metadata_root": str(metadata_root),
                    "processing": {
                        "selector": "linear-rulebase",
                        "scene_segments_dir": str(scene_root),
                    },
                },
                base_dir=root,
            )
            layout = LotLayout(data_root, "L21_a")
            layout.create_runtime_dirs()
            (layout.reports_dir / "videos.json").write_text(
                json.dumps({"videos": [{"video_id": "L21_V001"}]}),
                encoding="utf-8",
            )
            (metadata_root / "L21_V001.json").write_text("{\"fps\":25}", encoding="utf-8")
            (scene_root / "L21_V001.json").write_text(
                "{\"segments\":[{\"start_ms\":0,\"end_ms\":1000}]}",
                encoding="utf-8",
            )
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            orchestrator.shot_boundary_detector = object()
            orchestrator.embedding = None
            orchestrator._cached_runtime_signature = {"test": "runtime"}
            request = ArchiveInput(
                url="https://example.test/Videos_L21_a.zip",
                archive_name="Videos_L21_a.zip",
                lot_id="L21_a",
                line_number=1,
            )

            before = orchestrator._stage_fingerprint(request, "process_validate")
            before_scene = orchestrator._stage_fingerprint(request, "shot_boundaries")
            (metadata_root / "L22_V001.json").write_text("{\"fps\":30}", encoding="utf-8")
            (scene_root / "L22_V001.json").write_text(
                "{\"segments\":[{\"start_ms\":0,\"end_ms\":2000}]}",
                encoding="utf-8",
            )
            after_other_lot = orchestrator._stage_fingerprint(request, "process_validate")
            after_other_scene = orchestrator._stage_fingerprint(request, "shot_boundaries")
            (scene_root / "L21_V001.json").write_text(
                "{\"segments\":[{\"start_ms\":0,\"end_ms\":1500}]}",
                encoding="utf-8",
            )
            after_current_scene = orchestrator._stage_fingerprint(request, "process_validate")
            shot_after_current_scene = orchestrator._stage_fingerprint(request, "shot_boundaries")
            (metadata_root / "L21_V001.json").write_text("{\"fps\":24}", encoding="utf-8")
            after_current_lot = orchestrator._stage_fingerprint(request, "process_validate")

        self.assertEqual(before, after_other_lot)
        self.assertEqual(before_scene, after_other_scene)
        self.assertNotEqual(before, after_current_scene)
        self.assertEqual(before_scene, shot_after_current_scene)
        self.assertNotEqual(before, after_current_lot)

    def test_stage_runtime_signature_only_changes_for_relevant_source(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        config = BatchConfig()
        orchestrator = object.__new__(BatchOrchestrator)
        orchestrator.config = config
        changed = {"shot_boundaries.py": False}

        def fake_inventory(root, *, include=None, exclude_names=None):
            del root, exclude_names
            records = []
            for path in include or []:
                records.append(
                    {
                        "path": path.name,
                        "size_bytes": 1,
                        "sha256": (
                            "changed"
                            if path.name == "shot_boundaries.py"
                            and changed["shot_boundaries.py"]
                            else "stable"
                        ),
                    }
                )
            return records

        with patch("preprocess.batch.orchestrator.inventory", side_effect=fake_inventory), patch(
            "preprocess.batch.orchestrator.package_versions", return_value={}
        ), patch("preprocess.batch.orchestrator.tool_versions", return_value={}):
            before_shot = orchestrator._stage_runtime_signature("shot_boundaries")
            before_embedding = orchestrator._stage_runtime_signature("embedding")
            changed["shot_boundaries.py"] = True
            orchestrator._cached_stage_runtime_signatures = {}
            after_shot = orchestrator._stage_runtime_signature("shot_boundaries")
            after_embedding = orchestrator._stage_runtime_signature("embedding")

        self.assertNotEqual(before_shot, after_shot)
        self.assertEqual(before_embedding, after_embedding)

    def test_file_digest_cache_reuses_digest_until_stat_identity_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(b"first")
            cache = FileDigestCache(minimum_cache_bytes=0)
            with patch(
                "preprocess.batch.provenance.hashlib.sha256",
                wraps=hashlib.sha256,
            ) as sha_factory:
                first = cache.digest(path)
                second = cache.digest(path)
                previous = path.stat()
                path.write_bytes(b"other")
                os.utime(
                    path,
                    ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000),
                )
                third = cache.digest(path)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(sha_factory.call_count, 2)

    def test_file_fingerprint_uses_stat_then_hash_after_timestamp_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(b"payload")
            stat = path.stat()
            fingerprint = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
            with patch("preprocess.batch.provenance.sha256_file") as digest:
                self.assertTrue(file_fingerprint_matches(path, fingerprint))
                digest.assert_not_called()

            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            self.assertTrue(file_fingerprint_matches(path, fingerprint))

    def test_embedding_resumes_from_per_video_checkpoint(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        class FakeEmbedding:
            def __init__(self) -> None:
                self.calls = 0
                self.pipeline = type("Pipeline", (), {"cache_fingerprint": "fake-cache"})()

            def embed(self, layout, assets, results):
                del results
                self.calls += 1
                asset = assets[0]
                source_dir = layout.dataset_dir / "keyframes" / asset.video_id
                output_dir = layout.dataset_dir / "PECore-features" / asset.video_id
                output_dir.mkdir(parents=True, exist_ok=True)
                feature = output_dir / "000001.npy"
                np.save(feature, np.array([1, 0, 0, 0], dtype=np.float32))
                NpyFeatureWriter(4).write_video_metadata(
                    output_dir,
                    {
                        "000001": {
                            "schema_version": 1,
                            "image_sha256": sha256_file(source_dir / "000001.jpg"),
                            "cache_fingerprint": "fake-cache",
                        }
                    },
                )
                video = EmbeddingVideoResult(
                    asset.video_id,
                    source_dir,
                    output_dir,
                    1,
                    1,
                    0,
                    4,
                    (feature,),
                    "fake-cache",
                )
                return EmbeddingBatchResult((video,), 4, "fake-cache", {"fake": True})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping(
                {
                    "data_root": str(root / "data"),
                    "embedding": {"enabled": True, "expected_dim": 4},
                },
                base_dir=root,
            )
            layout = LotLayout(config.data_root, "L21_a")
            layout.create_runtime_dirs()
            source = layout.source_root / "L21_V001.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video")
            keyframes = layout.dataset_dir / "keyframes" / "L21_V001"
            keyframes.mkdir(parents=True)
            (keyframes / "000001.jpg").write_bytes(b"image")
            rendered = keyframes / "manifest.json"
            rendered.write_text("{}", encoding="utf-8")
            selection = layout.dataset_dir / "selection-manifests" / "L21_V001.json"
            selection.parent.mkdir(parents=True)
            selection.write_text("{}", encoding="utf-8")
            asset = VideoAsset("L21_V001", source, "L21_a", source.name)
            processed = ProcessingResult(asset, {}, 1, selection, rendered)
            request = ArchiveInput(
                "https://example.test/Videos_L21_a.zip",
                "Videos_L21_a.zip",
                "L21_a",
                1,
            )
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            orchestrator.embedding = FakeEmbedding()
            orchestrator.shot_boundary_detector = None
            orchestrator.progress = TqdmProgressReporter(ProgressConfig(enabled=False))
            orchestrator._cached_runtime_signature = {"test": "runtime"}
            checkpoints = CheckpointStore(layout.state_path)
            stage_fingerprint = orchestrator._stage_fingerprint(request, "embedding")

            checkpoints.start_stage("embedding", fingerprint=stage_fingerprint)
            first = orchestrator._embed(request, [asset], [processed], layout, checkpoints)
            checkpoints.complete_stage("embedding", fingerprint=stage_fingerprint)
            checkpoints.start_stage("embedding", fingerprint=stage_fingerprint)
            second = orchestrator._embed(request, [asset], [processed], layout, checkpoints)
            checkpoints.complete_stage("embedding", fingerprint=stage_fingerprint)
            feature_path = (
                layout.dataset_dir / "PECore-features" / asset.video_id / "000001.npy"
            )
            np.save(feature_path, np.zeros(4, dtype=np.float32))
            checkpoints.start_stage("embedding", fingerprint=stage_fingerprint)
            third = orchestrator._embed(request, [asset], [processed], layout, checkpoints)

        self.assertEqual(orchestrator.embedding.calls, 2)
        self.assertEqual(first.image_count, 1)
        self.assertEqual(second.image_count, 1)
        self.assertEqual(third.image_count, 1)

    def test_embedding_completion_journal_recovers_before_state_commit(self) -> None:
        from preprocess.batch.orchestrator import BatchOrchestrator

        class FakeEmbedding:
            def __init__(self) -> None:
                self.calls = 0
                self.pipeline = type(
                    "Pipeline", (), {"cache_fingerprint": "fake-cache"}
                )()

            def embed(self, layout, assets, results):
                del results
                self.calls += 1
                asset = assets[0]
                source_dir = layout.dataset_dir / "keyframes" / asset.video_id
                output_dir = (
                    layout.dataset_dir / "PECore-features" / asset.video_id
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                feature = output_dir / "000001.npy"
                np.save(feature, np.array([1, 0, 0, 0], dtype=np.float32))
                NpyFeatureWriter(4).write_video_metadata(
                    output_dir,
                    {
                        "000001": {
                            "schema_version": 1,
                            "image_sha256": sha256_file(
                                source_dir / "000001.jpg"
                            ),
                            "cache_fingerprint": "fake-cache",
                        }
                    },
                )
                result = EmbeddingVideoResult(
                    asset.video_id,
                    source_dir,
                    output_dir,
                    1,
                    1,
                    0,
                    4,
                    (feature,),
                    "fake-cache",
                )
                return EmbeddingBatchResult(
                    (result,), 4, "fake-cache", {"worker": "fake"}
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BatchConfig.from_mapping(
                {
                    "data_root": str(root / "data"),
                    "embedding": {"enabled": True, "expected_dim": 4},
                },
                base_dir=root,
            )
            layout = LotLayout(config.data_root, "L21_a")
            layout.create_runtime_dirs()
            source = layout.source_root / "L21_V001.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video")
            keyframes = layout.dataset_dir / "keyframes" / "L21_V001"
            keyframes.mkdir(parents=True)
            (keyframes / "000001.jpg").write_bytes(b"image")
            rendered = keyframes / "manifest.json"
            rendered.write_text("{}", encoding="utf-8")
            selection = (
                layout.dataset_dir
                / "selection-manifests"
                / "L21_V001.json"
            )
            selection.parent.mkdir(parents=True)
            selection.write_text("{}", encoding="utf-8")
            asset = VideoAsset("L21_V001", source, "L21_a", source.name)
            processed = ProcessingResult(asset, {}, 1, selection, rendered)
            orchestrator = object.__new__(BatchOrchestrator)
            orchestrator.config = config
            orchestrator.embedding = FakeEmbedding()
            checkpoints = CheckpointStore(layout.state_path)
            checkpoints.start_stage("embedding", fingerprint="embedding-stage")
            checkpoints.start_video(
                "embedding", asset.video_id, fingerprint="video-fingerprint"
            )

            orchestrator._run_background_embedding(
                orchestrator.embedding,
                layout,
                asset,
                processed,
                "video-fingerprint",
            )
            restored = orchestrator._restore_overlapped_embedding_video(
                asset,
                layout,
                checkpoints,
                "video-fingerprint",
                "fake-cache",
            )
            checkpoint_completed = checkpoints.video_is_complete(
                "embedding", asset.video_id, "video-fingerprint"
            )

        self.assertIsNotNone(restored)
        self.assertEqual(orchestrator.embedding.calls, 1)
        self.assertTrue(checkpoint_completed)

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
            feature_dir = layout.dataset_dir / "PECore-features" / "L21_V030"
            feature_dir.mkdir(parents=True)
            feature = feature_dir / "000000.npy"
            np.save(feature, np.ones(4, dtype=np.float32))
            (feature_dir / "provenance.json").write_text(
                '{"schema_version":1,"frames":{}}\n', encoding="utf-8"
            )
            (layout.reports_dir / "embedding.json").write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "video_id": "L21_V030",
                                "feature_files": [str(feature)],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
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
            metadata = json.loads(
                (staging.staging_dir / "dataset-metadata.json").read_text(encoding="utf-8")
            )
        self.assertIn("dataset-metadata.json", staged_paths)
        self.assertIn("keyframes/L21_V030/000000.jpg", staged_paths)
        self.assertIn("PECore-features/L21_V030/000000.npy", staged_paths)
        self.assertIn("PECore-features/L21_V030/provenance.json", staged_paths)
        self.assertIn("manifests/rendered/L21_V030.json", staged_paths)
        self.assertNotIn("videos/L21_V030.mp4", staged_paths)
        self.assertNotIn("source/L29_a/L21_V030.mp4", staged_paths)
        self.assertEqual(metadata["id"], "owner/test")

    def test_staging_includes_scene_segment_per_video_when_enabled(self) -> None:
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
            selection = layout.dataset_dir / "selection-manifests" / "L21_V030.json"
            selection.parent.mkdir(parents=True)
            selection.write_text("{}", encoding="utf-8")
            rendered = keyframes / "manifest.json"
            rendered.write_text("{}", encoding="utf-8")
            validation = layout.reports_dir / "validation" / "L21_V030.json"
            validation.parent.mkdir(parents=True)
            validation.write_text("{}", encoding="utf-8")
            source_video = layout.source_root / "L21_V030.mp4"
            source_video.parent.mkdir(parents=True)
            source_video.write_bytes(b"raw-video")
            scene_segments_dir = root / "scene-segments"
            scene_segments_dir.mkdir()
            scene_segments = scene_segments_dir / "L21_V030.json"
            scene_segments.write_text(
                '{"video_id":"L21_V030","segments":[{"start_ms":0,"end_ms":1000}]}\n',
                encoding="utf-8",
            )
            template = root / "dataset-metadata.json"
            template.write_text('{"title":"test"}\n', encoding="utf-8")

            asset = VideoAsset("L21_V030", source_video, "L29_a", source_video.name)
            result = ProcessingResult(asset, {"fps": None}, 1, selection, rendered)
            config = UploadConfig(
                enabled=True,
                dataset_ref="owner/test",
                metadata_template=template,
                include_scene_segments=True,
            )
            staging = KaggleStagingStrategy(
                JsonMetadataProvider(metadata_root),
                config,
                scene_segments_dir=scene_segments_dir,
            ).stage(layout, [asset], [result])
            staged_paths = {
                path.relative_to(staging.staging_dir).as_posix() for path in staging.files
            }

        self.assertIn("scene-segments/L21_V030.json", staged_paths)

    def test_staging_uses_rendered_manifest_allowlist_and_records_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_root = root / "metadata"
            metadata_root.mkdir()
            (metadata_root / "L21_V030.json").write_text("{}\n", encoding="utf-8")
            layout = LotLayout(root / "data", "L29_a")
            layout.create_runtime_dirs()
            keyframes = layout.dataset_dir / "keyframes" / "L21_V030"
            keyframes.mkdir(parents=True)
            (keyframes / "000000.jpg").write_bytes(b"selected")
            (keyframes / "999999.jpg").write_bytes(b"stale")
            rendered = keyframes / "manifest.json"
            rendered.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame": {"frame_id": "000000", "source_frame_number": 0},
                                "path": "000000.jpg",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            selection = layout.dataset_dir / "selection-manifests" / "L21_V030.json"
            selection.parent.mkdir(parents=True)
            selection.write_text("{}\n", encoding="utf-8")
            validation = layout.reports_dir / "validation" / "L21_V030.json"
            validation.parent.mkdir(parents=True)
            validation.write_text("{}\n", encoding="utf-8")
            source_video = layout.source_root / "L21_V030.mp4"
            source_video.parent.mkdir(parents=True)
            source_video.write_bytes(b"raw-video")
            template = root / "dataset-metadata.json"
            template.write_text('{"title":"test"}\n', encoding="utf-8")
            asset = VideoAsset("L21_V030", source_video, "L29_a", source_video.name)
            result = ProcessingResult(asset, {}, 1, selection, rendered)
            config = UploadConfig(
                enabled=True,
                dataset_ref="owner/test",
                metadata_template=template,
                include_features=False,
                include_transcripts=False,
                include_transcript_index=False,
            )
            staging = KaggleStagingStrategy(
                JsonMetadataProvider(metadata_root), config
            ).stage(layout, [asset], [result])
            staged_paths = {
                path.relative_to(staging.staging_dir).as_posix() for path in staging.files
            }
            digest, _ = digest_directory(
                staging.staging_dir,
                exclude_names={"provenance.json"},
            )
            provenance = json.loads(
                (staging.staging_dir / "provenance.json").read_text(encoding="utf-8")
            )

        self.assertIn("keyframes/L21_V030/000000.jpg", staged_paths)
        self.assertNotIn("keyframes/L21_V030/999999.jpg", staged_paths)
        self.assertEqual(staging.payload_digest, digest)
        self.assertEqual(provenance["payload_digest"], digest)

    def test_staging_error_policy_rejects_missing_feature_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_root = root / "metadata"
            metadata_root.mkdir()
            (metadata_root / "L21_V030.json").write_text("{}\n", encoding="utf-8")
            layout = LotLayout(root / "data", "L29_a")
            layout.create_runtime_dirs()
            keyframes = layout.dataset_dir / "keyframes" / "L21_V030"
            keyframes.mkdir(parents=True)
            (keyframes / "000000.jpg").write_bytes(b"selected")
            rendered = keyframes / "manifest.json"
            rendered.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame": {"frame_id": "000000", "source_frame_number": 0},
                                "path": "000000.jpg",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            selection = layout.dataset_dir / "selection-manifests" / "L21_V030.json"
            selection.parent.mkdir(parents=True)
            selection.write_text("{}\n", encoding="utf-8")
            validation = layout.reports_dir / "validation" / "L21_V030.json"
            validation.parent.mkdir(parents=True)
            validation.write_text("{}\n", encoding="utf-8")
            source_video = layout.source_root / "L21_V030.mp4"
            source_video.parent.mkdir(parents=True)
            source_video.write_bytes(b"raw-video")
            template = root / "dataset-metadata.json"
            template.write_text('{"title":"test"}\n', encoding="utf-8")
            asset = VideoAsset("L21_V030", source_video, "L29_a", source_video.name)
            result = ProcessingResult(asset, {}, 1, selection, rendered)
            config = UploadConfig(
                enabled=True,
                dataset_ref="owner/test",
                metadata_template=template,
                missing_artifact_policy="error",
                include_transcripts=False,
                include_transcript_index=False,
            )

            with self.assertRaisesRegex(FileNotFoundError, "Feature directory"):
                KaggleStagingStrategy(JsonMetadataProvider(metadata_root), config).stage(
                    layout, [asset], [result]
                )

            self.assertFalse(layout.staging_dir.exists())

    def test_cumulative_staging_keeps_previous_lot_and_replaces_only_same_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            metadata_root = root / "metadata"
            metadata_root.mkdir()
            template = root / "dataset-metadata.json"
            template.write_text('{"id":"owner/test","title":"test"}\n', encoding="utf-8")

            assets: list[VideoAsset] = []
            results: list[ProcessingResult] = []
            for lot_id, video_id in (("L21_a", "L21_V001"), ("L22_a", "L22_V001")):
                (metadata_root / f"{video_id}.json").write_text(
                    json.dumps({"video_id": video_id}) + "\n",
                    encoding="utf-8",
                )
                layout = LotLayout(data_root, lot_id)
                layout.create_runtime_dirs()
                keyframes = layout.dataset_dir / "keyframes" / video_id
                keyframes.mkdir(parents=True)
                (keyframes / "000000.jpg").write_bytes(video_id.encode("utf-8"))
                rendered = keyframes / "manifest.json"
                rendered.write_text("{}\n", encoding="utf-8")
                selection = layout.dataset_dir / "selection-manifests" / f"{video_id}.json"
                selection.parent.mkdir(parents=True)
                selection.write_text("{}\n", encoding="utf-8")
                validation = layout.reports_dir / "validation" / f"{video_id}.json"
                validation.parent.mkdir(parents=True)
                validation.write_text("{}\n", encoding="utf-8")
                source_video = layout.source_root / f"{video_id}.mp4"
                source_video.parent.mkdir(parents=True)
                source_video.write_bytes(b"placeholder-video")
                asset = VideoAsset(video_id, source_video, lot_id, source_video.name)
                assets.append(asset)
                results.append(
                    ProcessingResult(asset, {"fps": None}, 1, selection, rendered)
                )

            config = UploadConfig(
                enabled=True,
                dataset_ref="owner/test",
                staging_scope="dataset",
                metadata_template=template,
            )
            staging_dir = data_root / "kaggle-dataset-staging"
            state_store = DatasetUploadStateStore(data_root / "kaggle-dataset-state.json")
            strategy = CumulativeKaggleStagingStrategy(
                JsonMetadataProvider(metadata_root),
                config,
                staging_dir=staging_dir,
                state_store=state_store,
            )
            first_layout = LotLayout(data_root, "L21_a")
            second_layout = LotLayout(data_root, "L22_a")
            strategy.stage(first_layout, assets[:1], results[:1])
            strategy.stage(second_layout, assets[1:], results[1:])
            staged_paths = {
                path.relative_to(staging_dir).as_posix()
                for path in staging_dir.rglob("*")
                if path.is_file()
            }
            state = state_store.load()

        self.assertIn("keyframes/L21_V001/000000.jpg", staged_paths)
        self.assertIn("keyframes/L22_V001/000000.jpg", staged_paths)
        self.assertIn("metadata/L21_V001.json", staged_paths)
        self.assertIn("metadata/L22_V001.json", staged_paths)
        self.assertNotIn("source/L21_a/L21_V001.mp4", staged_paths)
        self.assertEqual(set(state["lots"]), {"L21_a", "L22_a"})
        self.assertEqual(state["lots"]["L22_a"]["status"], "staged")

    def test_cumulative_cleanup_preserves_dataset_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = LotLayout(root / "data", "L21_a")
            layout.create_runtime_dirs()
            cumulative_staging = root / "data" / "kaggle-dataset-staging"
            cumulative_staging.mkdir(parents=True)
            (cumulative_staging / "keyframes.zip").write_bytes(b"placeholder")
            upload = UploadResult("owner/test", "version", True, ("kaggle",), "ok")
            result = CleanupManager(
                CleanupConfig(enabled=True, delete_staging=True),
                preserve_staging=True,
            ).cleanup(layout, upload)
            staging_exists = cumulative_staging.is_dir()

        self.assertTrue(staging_exists)
        self.assertIn(
            "kaggle-staging: preserved for cumulative dataset",
            result.skipped,
        )

    def test_staging_requires_scene_segment_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_root = root / "metadata"
            metadata_root.mkdir()
            (metadata_root / "L21_V030.json").write_text("{}\n", encoding="utf-8")
            layout = LotLayout(root / "data", "L29_a")
            layout.create_runtime_dirs()
            keyframes = layout.dataset_dir / "keyframes" / "L21_V030"
            keyframes.mkdir(parents=True)
            (keyframes / "000000.jpg").write_bytes(b"jpg")
            selection = layout.dataset_dir / "selection-manifests" / "L21_V030.json"
            selection.parent.mkdir(parents=True)
            selection.write_text("{}", encoding="utf-8")
            rendered = keyframes / "manifest.json"
            rendered.write_text("{}", encoding="utf-8")
            source_video = layout.source_root / "L21_V030.mp4"
            source_video.parent.mkdir(parents=True)
            source_video.write_bytes(b"raw-video")
            template = root / "dataset-metadata.json"
            template.write_text('{"title":"test"}\n', encoding="utf-8")
            asset = VideoAsset("L21_V030", source_video, "L29_a", source_video.name)
            result = ProcessingResult(asset, {"fps": None}, 1, selection, rendered)
            config = UploadConfig(
                enabled=True,
                dataset_ref="owner/test",
                metadata_template=template,
                include_scene_segments=True,
            )

            with self.assertRaises(FileNotFoundError):
                KaggleStagingStrategy(
                    JsonMetadataProvider(metadata_root),
                    config,
                    scene_segments_dir=root / "scene-segments",
                ).stage(layout, [asset], [result])

            self.assertFalse(layout.staging_dir.exists())

    def test_kaggle_upload_command_uses_configured_zip_dir_mode(self) -> None:
        config = UploadConfig(
            enabled=True,
            dataset_ref="owner/test",
            mode="version",
            version_message="test upload",
            dir_mode="zip",
        )
        command = KaggleCliUploader("kaggle", config)._upload_command(Path("staging"))
        self.assertEqual(
            command,
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                "staging",
                "-m",
                "test upload",
                "--dir-mode",
                "zip",
            ],
        )

    def test_kaggle_status_verify_uses_cli_compatible_command(self) -> None:
        config = UploadConfig(
            enabled=True,
            dataset_ref="owner/test",
            verify_timeout_seconds=1,
            verify_poll_seconds=1,
        )
        uploader = KaggleCliUploader("kaggle", config)
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "ready\n", "stderr": ""},
        )()

        with patch(
            "preprocess.batch.kaggle_uploader.subprocess.run",
            return_value=completed,
        ) as run:
            with patch(
                "preprocess.batch.kaggle_uploader.time.monotonic",
                side_effect=[0.0, 0.1],
            ):
                verified, output = uploader._verify()

        self.assertTrue(verified)
        self.assertEqual(output, "ready\n")
        run.assert_called_once_with(
            ["kaggle", "datasets", "status", "owner/test"],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_kaggle_auto_mode_creates_missing_lot_dataset(self) -> None:
        config = UploadConfig(
            enabled=True,
            dataset_ref_template="owner/aic2026-hcmc-{lot_slug}",
            mode="auto",
            public=False,
        )
        uploader = KaggleCliUploader("kaggle", config)
        target = config.target_for_lot("L22_a")
        staging = StagingResult(
            staging_dir=Path("staging"),
            files=(),
            target=target,
        )
        missing = type(
            "Completed",
            (),
            {"returncode": 1, "stdout": "", "stderr": "not found"},
        )()
        ready = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "ready", "stderr": ""},
        )()

        with patch(
            "preprocess.batch.kaggle_uploader.subprocess.run",
            side_effect=[missing, ready],
        ):
            with patch.object(
                uploader,
                "_run_streaming",
                return_value=(0, "created"),
            ) as upload_command:
                result = uploader.upload_and_verify(staging)

        self.assertEqual(result.dataset_ref, "owner/aic2026-hcmc-l22-a")
        self.assertEqual(result.mode, "create")
        self.assertEqual(upload_command.call_args.args[0][:4], [
            "kaggle",
            "datasets",
            "create",
            "-p",
        ])
        self.assertNotIn("--private", upload_command.call_args.args[0])

    def test_kaggle_auto_mode_treats_legacy_forbidden_status_as_create(self) -> None:
        config = UploadConfig(
            enabled=True,
            dataset_ref_template="owner/aic-{lot_slug}",
            mode="auto",
        )
        uploader = KaggleCliUploader("kaggle", config)
        forbidden = type(
            "Completed",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "403 Client Error: Forbidden for url: status/owner/aic-l25-a",
            },
        )()
        with patch(
            "preprocess.batch.kaggle_uploader.subprocess.run",
            return_value=forbidden,
        ):
            mode = uploader._resolve_mode(config.target_for_lot("L25_a"))
        self.assertEqual(mode, "create")

    def test_kaggle_evidence_requires_remote_provenance_and_local_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            (staging / "keyframes.zip").write_bytes(b"payload")
            digest, _ = digest_directory(staging)
            atomic_json_write(staging / "provenance.json", {"payload_digest": digest})
            config = UploadConfig(
                enabled=True,
                dataset_ref="owner/test",
                verify_timeout_seconds=1,
                verify_poll_seconds=1,
                require_remote_inventory=True,
            )
            uploader = KaggleCliUploader("kaggle", config)
            status = type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ready\n", "stderr": ""},
            )()
            files = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "name size creationDate\nprovenance.json 12 today\n",
                    "stderr": "",
                },
            )()
            with patch(
                "preprocess.batch.kaggle_uploader.subprocess.run",
                side_effect=[status, files],
            ):
                evidence = uploader._verify_evidence(
                    expected_payload_digest=digest,
                    provenance_path=staging / "provenance.json",
                )

        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["status"], "ready")
        self.assertIn("provenance.json", evidence["files"])

    def test_kaggle_local_audit_rejects_inventory_path_outside_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            (root / "secret.bin").write_bytes(b"not-a-staging-file")
            digest = "0" * 64
            atomic_json_write(
                staging / "provenance.json",
                {
                    "payload_digest": digest,
                    "files": [
                        {
                            "path": "../secret.bin",
                            "size_bytes": 18,
                            "sha256": sha256_file(root / "secret.bin"),
                        }
                    ],
                },
            )
            uploader = KaggleCliUploader(
                "kaggle",
                UploadConfig(
                    enabled=True,
                    dataset_ref="owner/test",
                    verify_timeout_seconds=1,
                    verify_poll_seconds=1,
                ),
            )
            status = type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ready\n", "stderr": ""},
            )()
            files = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "name size creationDate\nprovenance.json 12 today\n",
                    "stderr": "",
                },
            )()
            with patch(
                "preprocess.batch.kaggle_uploader.subprocess.run",
                side_effect=[status, files],
            ):
                evidence = uploader._verify_evidence(
                    expected_payload_digest=digest,
                    provenance_path=staging / "provenance.json",
                    audit_local_payload=True,
                )

        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["status"], "local_payload_changed")

    def test_cumulative_transaction_recovery_publishes_ready_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            staging_dir = data_root / "kaggle-dataset-staging"
            staging_dir.mkdir(parents=True)
            (staging_dir / "old.txt").write_text("old", encoding="utf-8")
            candidate = data_root / ".kaggle-dataset-staging.candidate.test"
            candidate.mkdir(parents=True)
            (candidate / "new.txt").write_text("new", encoding="utf-8")
            backup = data_root / ".kaggle-dataset-staging.backup.test"
            os.replace(staging_dir, backup)
            marker = data_root / "kaggle-dataset-transaction.json"
            atomic_json_write(
                marker,
                {
                    "status": "ready",
                    "candidate_dir": str(candidate),
                    "backup_dir": str(backup),
                },
            )
            strategy = object.__new__(CumulativeKaggleStagingStrategy)
            strategy.staging_dir = staging_dir
            strategy._recover_transaction()

            self.assertEqual((staging_dir / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(backup.exists())
            self.assertFalse(marker.exists())

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
            self.assertFalse(layout.staging_dir.exists())
            self.assertTrue(metadata.is_file())


if __name__ == "__main__":
    unittest.main()
