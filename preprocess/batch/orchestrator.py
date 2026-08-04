"""Dependency-injected orchestration for one or more archive-derived lots."""
from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

from preprocess.batch.archive_extractor import ArchiveExtractor, ZipArchiveExtractor
from preprocess.batch.archive_validator import ZipArchiveValidator
from preprocess.batch.checkpoints import CheckpointStore
from preprocess.batch.config import BatchConfig, selector_requires_scene_boundaries
from preprocess.batch.cleanup import CleanupManager
from preprocess.batch.dataset_state import (
    DatasetUploadStateStore,
    dataset_state_path,
    dataset_upload_lock_path,
)
from preprocess.batch.downloader import Aria2ArchiveDownloader, ArchiveDownloader
from preprocess.batch.embedding import (
    BatchEmbeddingStrategy,
    default_pecore_embedding_strategy,
)
from preprocess.batch.layout import LotLayout
from preprocess.batch.locks import ExclusiveFileLock
from preprocess.batch.links import LinkListParser
from preprocess.batch.metadata import JsonMetadataProvider, MetadataProvider
from preprocess.batch.models import (
    ArchiveInput,
    ArchiveInspection,
    BatchState,
    ProcessingResult,
    UploadResult,
    VideoAsset,
    utc_now,
)
from preprocess.batch.kaggle_uploader import (
    DatasetUploader,
    KaggleCliUploader,
    CumulativeKaggleStagingStrategy,
    KaggleStagingStrategy,
    StagingStrategy,
)
from preprocess.batch.preflight import PreflightChecker
from preprocess.batch.processor import (
    BatchProcessor,
    KeyframeProcessingStrategy,
    ProcessingStrategyRegistry,
    SelectionStrategyRegistry,
    default_processing_registry,
    default_selection_registry,
)
from preprocess.batch.shot_boundaries import (
    ShotBoundaryArtifact,
    ShotBoundaryDetector,
    ShotBoundaryPipeline,
    default_shot_boundary_registry,
    load_scene_segments,
)
from preprocess.pecore.embedding import EmbeddingBatchResult, EmbeddingVideoResult, NpyFeatureWriter
from preprocess.progress import ProgressReporter, TqdmProgressReporter
from preprocess.batch.validators import (
    VideoValidationContext,
    VideoValidationPipeline,
    default_video_validation_pipeline,
    load_video_metadata,
)


@dataclass(frozen=True)
class LotRunResult:
    lot_id: str
    assets: tuple[VideoAsset, ...]
    processed: tuple[ProcessingResult, ...]
    upload: UploadResult | None
    embedding: EmbeddingBatchResult | None = None


class BatchOrchestrator:
    """Coordinate stages while keeping every external dependency injectable."""

    def __init__(
        self,
        config: BatchConfig,
        *,
        downloader: ArchiveDownloader,
        archive_validator: ZipArchiveValidator,
        archive_extractor: ArchiveExtractor,
        metadata_provider: MetadataProvider,
        processor: BatchProcessor,
        video_validator: VideoValidationPipeline,
        stager: StagingStrategy | None,
        uploader: DatasetUploader | None,
        cleanup: CleanupManager,
        shot_boundary_detector: ShotBoundaryDetector | None = None,
        embedding: BatchEmbeddingStrategy | None = None,
        progress: ProgressReporter | None = None,
        dataset_upload_state: DatasetUploadStateStore | None = None,
        dataset_upload_lock_path: Path | None = None,
    ) -> None:
        self.config = config
        self.downloader = downloader
        self.archive_validator = archive_validator
        self.archive_extractor = archive_extractor
        self.metadata_provider = metadata_provider
        self.processor = processor
        self.video_validator = video_validator
        self.stager = stager
        self.uploader = uploader
        self.cleanup = cleanup
        self.shot_boundary_detector = shot_boundary_detector
        self.embedding = embedding
        self.progress = progress or TqdmProgressReporter(config.progress)
        self.dataset_upload_state = dataset_upload_state
        self.dataset_upload_lock_path = dataset_upload_lock_path

    def run_all(self, requests: Sequence[ArchiveInput]) -> list[LotRunResult]:
        stage_names = self._pipeline_stage_names()
        completed_flags = [self._is_completed_lot(request) for request in requests]
        if not requests or not all(completed_flags):
            PreflightChecker(self.config).run()
        self.progress.start_pipeline(
            total_units=len(requests) * len(stage_names),
            total_lots=len(requests),
            stage_names=stage_names,
        )
        results: list[LotRunResult] = []
        try:
            for lot_index, request in enumerate(requests, start=1):
                self.progress.set_lot_context(
                    lot_index=lot_index,
                    total_lots=len(requests),
                    lot_id=request.lot_id,
                )
                if completed_flags[lot_index - 1]:
                    self.progress.skip_lot(
                        lot_id=request.lot_id,
                        stage_count=len(stage_names),
                    )
                    continue
                results.append(self.run_lot(request))
        finally:
            self.progress.finish_pipeline()
        return results

    def _is_completed_lot(self, request: ArchiveInput) -> bool:
        """Return true when a lot is terminal and still represents this request."""
        layout = LotLayout(self.config.data_root, request.lot_id)
        state = CheckpointStore(layout.state_path).load()
        if state.get("state") != BatchState.COMPLETED.value:
            return False
        previous_request = state.get("request")
        if previous_request is not None and previous_request != request.to_dict():
            raise RuntimeError(
                f"Lot is already completed for a different archive request: {request.lot_id}. "
                "Use a new lot directory for a different URL or archive name."
            )
        return True

    def _pipeline_stage_names(self) -> tuple[str, ...]:
        stages = ["download", "archive_validate", "extract", "discover"]
        if self.shot_boundary_detector is not None:
            stages.append("shot_boundaries")
        stages.append("process_validate")
        if self.embedding is not None:
            stages.append("embedding")
        if self.config.upload.enabled:
            stages.append("stage_upload")
            stages.append("cleanup")
        return tuple(stages)

    def run_lot(self, request: ArchiveInput) -> LotRunResult:
        layout = LotLayout(self.config.data_root, request.lot_id)
        layout.create_runtime_dirs()
        checkpoints = CheckpointStore(layout.state_path)
        self._initialize_state(checkpoints, request)
        upload_checkpoints: CheckpointStore | None = None
        try:
            archive_path = self._execute_stage(
                checkpoints,
                request,
                "download",
                action=lambda: self._download(request, layout, checkpoints),
                restore=lambda: self._restore_archive(request, layout),
            )
            inspection = self._execute_stage(
                checkpoints,
                request,
                "archive_validate",
                action=lambda: self._validate_archive(archive_path, layout, checkpoints),
                restore=lambda: self._restore_archive_inspection(archive_path),
            )
            self._execute_stage(
                checkpoints,
                request,
                "extract",
                action=lambda: self._extract(inspection, request, layout, checkpoints),
                restore=lambda: self._restore_extraction(inspection, layout),
            )
            assets = self._execute_stage(
                checkpoints,
                request,
                "discover",
                action=lambda: self._discover(layout, request, checkpoints),
                restore=lambda: self._restore_assets(layout, request),
            )

            if self.shot_boundary_detector is not None:
                self._execute_stage(
                    checkpoints,
                    request,
                    "shot_boundaries",
                    action=lambda: self._detect_shot_boundaries(assets, layout, checkpoints),
                    restore=lambda: self._restore_shot_boundaries(assets),
                )

            processed = self._execute_stage(
                checkpoints,
                request,
                "process_validate",
                action=lambda: self._process_and_validate(request, assets, layout, checkpoints),
                restore=lambda: self._restore_processed(assets, layout),
            )

            if self.embedding is not None:
                embedding = self._execute_stage(
                    checkpoints,
                    request,
                    "embedding",
                    action=lambda: self._embed(assets, processed, layout, checkpoints),
                    restore=lambda: self._restore_embedding(layout),
                )
            else:
                embedding = None

            if self.config.upload.enabled:
                with self._upload_locks(layout, request.lot_id):
                    upload_checkpoints = self._initialize_upload_state(
                        CheckpointStore(layout.upload_state_path),
                        request,
                        legacy_state=checkpoints.load(),
                    )
                    upload = self._run_upload_stages(
                        request,
                        assets,
                        processed,
                        layout,
                        upload_checkpoints,
                    )
            else:
                upload = None
            checkpoints.transition(BatchState.COMPLETED, payload={"finished_at": utc_now()})
            return LotRunResult(request.lot_id, tuple(assets), tuple(processed), upload, embedding)
        except Exception as exc:
            checkpoints.transition(
                BatchState.FAILED,
                payload={"error": f"{type(exc).__name__}: {exc}", "failed_at": utc_now()},
            )
            if upload_checkpoints is not None:
                upload_checkpoints.transition(
                    BatchState.FAILED,
                    payload={"error": f"{type(exc).__name__}: {exc}", "failed_at": utc_now()},
                )
            raise

    def upload_lot(self, lot_id: str) -> UploadResult:
        """Explicitly upload one lot, independently of the full-run flag."""
        layout = LotLayout(self.config.data_root, lot_id)
        if not layout.state_path.is_file():
            raise FileNotFoundError(f"Main checkpoint not found: {layout.state_path}")

        main_checkpoints = CheckpointStore(layout.state_path)
        main_state = main_checkpoints.load()
        request = self._request_from_state(main_state, lot_id)

        self.progress.start_pipeline(
            total_units=2,
            total_lots=1,
            stage_names=("stage_upload", "cleanup"),
        )
        self.progress.set_lot_context(lot_index=1, total_lots=1, lot_id=lot_id)
        upload_checkpoints: CheckpointStore | None = None
        try:
            with self._upload_locks(layout, lot_id):
                upload_checkpoints = self._initialize_upload_state(
                    CheckpointStore(layout.upload_state_path),
                    request,
                    legacy_state=main_state,
                )
                if self._upload_state_is_complete(upload_checkpoints):
                    result = self._restore_upload(layout)
                    if upload_checkpoints.load().get("state") != BatchState.COMPLETED.value:
                        upload_checkpoints.transition(
                            BatchState.COMPLETED,
                            payload={"finished_at": utc_now(), "reused": True},
                        )
                else:
                    assets = self._restore_assets(layout, request)
                    processed = self._restore_processed(assets, layout)
                    required_stage = "embedding" if self.embedding is not None else "process_validate"
                    stage_record = main_state.get("stages", {}).get(required_stage, {})
                    if not isinstance(stage_record, Mapping) or stage_record.get("status") != "completed":
                        raise RuntimeError(
                            f"Cannot upload {lot_id}: required stage {required_stage!r} is not completed"
                        )
                    result = self._run_upload_stages(
                        request,
                        assets,
                        processed,
                        layout,
                        upload_checkpoints,
                        reuse_completed=True,
                    )
            main_checkpoints.transition(
                BatchState.COMPLETED,
                payload={"finished_at": utc_now(), "upload_only": True},
            )
            return result
        except Exception as exc:
            if upload_checkpoints is not None:
                upload_checkpoints.transition(
                    BatchState.FAILED,
                    payload={"error": f"{type(exc).__name__}: {exc}", "failed_at": utc_now()},
                )
            raise
        finally:
            self.progress.finish_pipeline()

    @contextmanager
    def _upload_locks(self, layout: LotLayout, lot_id: str) -> Iterator[None]:
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                ExclusiveFileLock(layout.upload_lock_path, purpose=f"upload for {lot_id}")
            )
            if self.config.upload.staging_scope == "dataset":
                dataset_lock_path = getattr(self, "dataset_upload_lock_path", None)
                if dataset_lock_path is None:
                    dataset_lock_path = self.config.data_root / "kaggle-dataset-upload.lock"
                stack.enter_context(
                    ExclusiveFileLock(
                        dataset_lock_path,
                        purpose="cumulative dataset upload",
                    )
                )
            yield

    @staticmethod
    def _upload_state_is_complete(checkpoints: CheckpointStore) -> bool:
        state = checkpoints.load()
        stages = state.get("stages", {})
        if not isinstance(stages, Mapping):
            return False
        return all(
            isinstance(stages.get(name), Mapping)
            and stages[name].get("status") == "completed"
            for name in ("stage_upload", "cleanup")
        )

    def _initialize_upload_state(
        self,
        checkpoints: CheckpointStore,
        request: ArchiveInput,
        *,
        legacy_state: Mapping[str, Any] | None = None,
    ) -> CheckpointStore:
        state = checkpoints.load()
        if not state.get("initialized"):
            migrated_stages: dict[str, Any] = {}
            if isinstance(legacy_state, Mapping):
                legacy_stages = legacy_state.get("stages", {})
                if isinstance(legacy_stages, Mapping):
                    for name in ("stage_upload", "cleanup"):
                        value = legacy_stages.get(name)
                        if isinstance(value, Mapping):
                            migrated_stages[name] = dict(value)
            state = {
                "state": BatchState.NEW.value,
                "initialized": True,
                "resume_version": 1,
                "lot_id": request.lot_id,
                "request": request.to_dict(),
                "config": self.config.to_dict(),
                "config_fingerprint": self._config_fingerprint(),
                "request_fingerprint": self._request_fingerprint(request),
                "created_at": utc_now(),
                "events": [],
                "stages": migrated_stages,
            }
            state["migrated_from_state"] = legacy_state is not None
            checkpoints.write(state)
            return checkpoints

        previous_request = state.get("request")
        if previous_request is not None and previous_request != request.to_dict():
            raise RuntimeError(
                f"Archive request changed after upload started: {request.lot_id}."
            )
        state["resume_version"] = 1
        state["config"] = self.config.to_dict()
        state["config_fingerprint"] = self._config_fingerprint()
        state["request_fingerprint"] = self._request_fingerprint(request)
        state.setdefault("request", request.to_dict())
        state.setdefault("stages", {})
        checkpoints.write(state)
        return checkpoints

    @staticmethod
    def _request_from_state(state: Mapping[str, Any], lot_id: str) -> ArchiveInput:
        raw_request = state.get("request")
        if not isinstance(raw_request, Mapping):
            raise ValueError(f"Main checkpoint has no archive request for {lot_id}")
        try:
            request = ArchiveInput(
                url=str(raw_request["url"]),
                archive_name=str(raw_request["archive_name"]),
                lot_id=str(raw_request["lot_id"]),
                line_number=int(raw_request["line_number"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Main checkpoint has an invalid archive request for {lot_id}") from exc
        if request.lot_id != lot_id:
            raise ValueError(
                f"Main checkpoint lot mismatch: expected {lot_id}, got {request.lot_id}"
            )
        return request

    def _initialize_state(self, checkpoints: CheckpointStore, request: ArchiveInput) -> None:
        state = checkpoints.load()
        if state.get("state") == BatchState.COMPLETED.value:
            raise RuntimeError(f"Lot is already completed: {request.lot_id}")
        if not state.get("initialized"):
            state = {
                "state": BatchState.NEW.value,
                "initialized": True,
                "resume_version": 1,
                "lot_id": request.lot_id,
                "request": request.to_dict(),
                "config": self.config.to_dict(),
                "config_fingerprint": self._config_fingerprint(),
                "request_fingerprint": self._request_fingerprint(request),
                "created_at": utc_now(),
                "events": [],
                "stages": {},
            }
            checkpoints.write(state)
            return

        previous_request = state.get("request")
        if previous_request is not None and previous_request != request.to_dict():
            raise RuntimeError(
                f"Archive request changed after this lot started: {request.lot_id}. "
                "Use a new lot directory for a different URL or archive name."
            )

        # Checkpoints written by the pre-stage-resume version remain usable;
        # their artifact caches are still validated by each restore method.
        state["resume_version"] = 1
        state["config_fingerprint"] = self._config_fingerprint()
        state["request_fingerprint"] = self._request_fingerprint(request)
        state.setdefault("request", request.to_dict())
        state.setdefault("stages", {})
        checkpoints.write(state)

    def _config_fingerprint(self) -> str:
        encoded = json.dumps(
            self.config.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _request_fingerprint(request: ArchiveInput) -> str:
        encoded = json.dumps(
            request.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _stage_fingerprint(self, request: ArchiveInput, name: str) -> str:
        config = self.config.to_dict()
        stage_config: dict[str, Any]
        if name == "download":
            stage_config = {
                "tools": config["tools"],
                "download": config["download"],
                "archive": config["archive"],
                "minimum_free_bytes": config["minimum_free_bytes"],
            }
        elif name == "archive_validate":
            stage_config = {"archive": config["archive"]}
        elif name == "extract":
            stage_config = {"archive": config["archive"]}
        elif name == "discover":
            stage_config = {"archive": config["archive"]}
        elif name == "shot_boundaries":
            stage_config = {
                "shot_boundary": config["shot_boundary"],
                "scene_segments_dir": config["processing"]["scene_segments_dir"],
            }
        elif name == "process_validate":
            stage_config = {
                "archive": config["archive"],
                "processing": config["processing"],
                "metadata_root": config["metadata_root"],
                "scene_boundaries": config["shot_boundary"],
            }
        elif name == "embedding":
            stage_config = {
                "embedding": config["embedding"],
                "profile_id": config["processing"]["profile_id"],
            }
        elif name == "stage_upload":
            stage_config = {
                "upload": config["upload"],
                "profile_id": config["processing"]["profile_id"],
            }
        elif name == "cleanup":
            stage_config = {
                "cleanup": config["cleanup"],
                "profile_id": config["processing"]["profile_id"],
            }
        else:
            raise ValueError(f"Unknown pipeline stage: {name}")

        dependencies: list[str] = []
        if name == "archive_validate":
            dependencies.append("download")
        elif name == "extract":
            dependencies.append("archive_validate")
        elif name == "discover":
            dependencies.append("extract")
        elif name == "shot_boundaries":
            dependencies.append("discover")
        elif name == "process_validate":
            dependencies.append("shot_boundaries" if self.shot_boundary_detector is not None else "discover")
        elif name == "embedding":
            dependencies.append("process_validate")
        elif name == "stage_upload":
            dependencies.append("embedding" if self.embedding is not None else "process_validate")
        elif name == "cleanup":
            dependencies.append("stage_upload")

        payload = {
            "request": request.to_dict(),
            "stage": name,
            "config": stage_config,
            "dependencies": [self._stage_fingerprint(request, dependency) for dependency in dependencies],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _video_checkpoint_fingerprint(stage_fingerprint: str, asset: VideoAsset) -> str:
        """Fingerprint one source video within a process/validate stage."""
        stat = asset.path.stat()
        payload = {
            "stage": stage_fingerprint,
            "video_id": asset.video_id,
            "path": str(asset.path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _execute_stage(
        self,
        checkpoints: CheckpointStore,
        request: ArchiveInput,
        name: str,
        *,
        action: Callable[[], Any],
        restore: Callable[[], Any],
        reuse_completed: bool = False,
    ) -> Any:
        self.progress.start_stage(name=name, lot_id=request.lot_id)
        fingerprint = self._stage_fingerprint(request, name)
        stage_completed = checkpoints.stage_is_complete(name, fingerprint)
        if reuse_completed:
            stage = checkpoints.load().get("stages", {}).get(name, {})
            stage_completed = stage_completed or (
                isinstance(stage, Mapping) and stage.get("status") == "completed"
            )
        if stage_completed:
            try:
                value = restore()
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                checkpoints.invalidate_stage(name, reason=f"artifact restore failed: {exc}")
            else:
                self.progress.complete_stage(name=name, lot_id=request.lot_id)
                return value

        checkpoints.start_stage(name, fingerprint=fingerprint)
        value = action()
        checkpoints.complete_stage(name, fingerprint=fingerprint)
        self.progress.complete_stage(name=name, lot_id=request.lot_id)
        return value

    def _restore_archive(self, request: ArchiveInput, layout: LotLayout) -> Path:
        path = layout.archive_dir / request.archive_name
        self.archive_validator.validate(path)
        return path

    def _restore_archive_inspection(self, archive_path: Path) -> ArchiveInspection:
        return self.archive_validator.validate(archive_path)

    def _restore_extraction(self, inspection: ArchiveInspection, layout: LotLayout) -> Path:
        source_root = layout.source_root
        if not source_root.is_dir():
            raise FileNotFoundError(f"Extracted source root not found: {source_root}")
        for member in inspection.video_members:
            parts = PurePosixPath(member.replace("\\", "/")).parts
            if not parts or parts[0] != inspection.root_name:
                raise ValueError(f"Archive member is outside expected root: {member}")
            extracted = source_root.joinpath(*parts[1:])
            if not extracted.is_file() or extracted.stat().st_size <= 0:
                raise FileNotFoundError(f"Extracted video is missing or empty: {extracted}")
        return source_root

    def _restore_assets(self, layout: LotLayout, request: ArchiveInput) -> list[VideoAsset]:
        report_path = layout.reports_dir / "videos.json"
        payload = self._read_json(report_path)
        records = payload.get("videos")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Video discovery report is empty or invalid: {report_path}")
        assets: list[VideoAsset] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Invalid video discovery record: {report_path}")
            asset = VideoAsset(
                video_id=str(record["video_id"]),
                path=Path(str(record["path"])),
                lot_id=str(record["lot_id"]),
                source_name=str(record["source_name"]),
            )
            if asset.lot_id != request.lot_id or not asset.path.is_file():
                raise FileNotFoundError(f"Discovered source video is no longer available: {asset.path}")
            try:
                asset.path.resolve().relative_to(layout.source_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Discovered video is outside the lot source root: {asset.path}"
                ) from exc
            assets.append(asset)
        if len({asset.video_id for asset in assets}) != len(assets):
            raise ValueError(f"Duplicate video IDs in discovery report: {report_path}")
        return assets

    def _restore_shot_boundaries(self, assets: Sequence[VideoAsset]) -> tuple[ShotBoundaryArtifact, ...]:
        if self.shot_boundary_detector is None:
            return ()
        output_dir = self.config.processing.scene_segments_dir
        if output_dir is None:
            raise ValueError("Scene-boundary output directory is not configured")
        restored: list[ShotBoundaryArtifact] = []
        for asset in assets:
            path = output_dir / f"{asset.video_id}.json"
            payload = self._read_json(path)
            if str(payload.get("backend", "")) != self.shot_boundary_detector.name:
                raise ValueError(f"Shot-boundary backend changed for {asset.video_id}: {path}")
            expected_threshold = getattr(getattr(self.shot_boundary_detector, "config", None), "threshold", None)
            if expected_threshold is not None and float(payload.get("threshold")) != float(expected_threshold):
                raise ValueError(f"Shot-boundary threshold changed for {asset.video_id}: {path}")
            source = payload.get("source")
            if isinstance(source, Mapping) and isinstance(source.get("fingerprint"), Mapping):
                fingerprint = source["fingerprint"]
                stat = asset.path.stat()
                if (
                    int(fingerprint.get("size_bytes")) != stat.st_size
                    or int(fingerprint.get("mtime_ns")) != stat.st_mtime_ns
                ):
                    raise ValueError(f"Source video changed for shot-boundary artifact: {asset.video_id}")
            segments = load_scene_segments(path)
            restored.append(
                ShotBoundaryArtifact(
                    video_id=asset.video_id,
                    path=path,
                    backend=self.shot_boundary_detector.name,
                    scene_count=len(segments),
                    cached=True,
                )
            )
        return tuple(restored)

    def _restore_processed(
        self,
        assets: Sequence[VideoAsset],
        layout: LotLayout,
    ) -> list[ProcessingResult]:
        report_path = layout.reports_dir / "processing.json"
        payload = self._read_json(report_path)
        records = payload.get("videos")
        if not isinstance(records, list):
            raise ValueError(f"Processing report is invalid: {report_path}")
        asset_by_id = {asset.video_id: asset for asset in assets}
        record_by_id: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Invalid processing record: {report_path}")
            raw_asset = record.get("asset")
            if not isinstance(raw_asset, Mapping):
                raise ValueError(f"Processing record has no asset: {report_path}")
            video_id = str(raw_asset.get("video_id", ""))
            if video_id not in asset_by_id:
                raise ValueError(f"Processing report contains an unknown video: {video_id}")
            record_by_id[video_id] = record
        if set(record_by_id) != set(asset_by_id):
            raise ValueError(f"Processing report does not cover all videos: {report_path}")
        return [
            self._restore_processed_video(asset, layout, record_by_id[asset.video_id])
            for asset in assets
        ]

    def _restore_processed_video(
        self,
        asset: VideoAsset,
        layout: LotLayout,
        record: Mapping[str, Any],
    ) -> ProcessingResult:
        """Validate and restore one video's processing result and artifacts."""
        raw_asset = record.get("asset")
        if not isinstance(raw_asset, Mapping) or str(raw_asset.get("video_id", "")) != asset.video_id:
            raise ValueError(f"Processing record does not match video: {asset.video_id}")
        video_id = asset.video_id
        selection_path = Path(str(record["selection_manifest_path"]))
        rendered_path = Path(str(record["rendered_manifest_path"]))
        validation_path = layout.reports_dir / "validation" / f"{video_id}.json"
        if not selection_path.is_file() or not rendered_path.is_file() or not validation_path.is_file():
            raise FileNotFoundError(f"Processing artifacts are incomplete for {video_id}")
        selection = self._read_json(selection_path)
        source = selection.get("source")
        fingerprint = source.get("fingerprint") if isinstance(source, Mapping) else None
        if isinstance(fingerprint, Mapping):
            stat = asset.path.stat()
            if (
                int(fingerprint.get("size_bytes")) != stat.st_size
                or int(fingerprint.get("mtime_ns")) != stat.st_mtime_ns
            ):
                raise ValueError(f"Source video changed after processing: {video_id}")
        validation = self._read_json(validation_path)
        if validation.get("passed") is not True:
            raise ValueError(f"Validation report is not passing for {video_id}")
        rendered = self._read_json(rendered_path)
        frames = rendered.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"Rendered manifest is invalid for {video_id}")
        for frame in frames:
            if not isinstance(frame, Mapping) or not frame.get("path"):
                raise ValueError(f"Rendered frame record is invalid for {video_id}")
            frame_path = Path(str(frame["path"]))
            candidates = [frame_path]
            if not frame_path.is_absolute():
                candidates.append(rendered_path.parent / frame_path.name)
            if not any(candidate.is_file() for candidate in candidates):
                raise FileNotFoundError(f"Rendered frame is missing for {video_id}: {frame_path}")
        return ProcessingResult(
            asset=asset,
            video_info=dict(record.get("video_info", {})),
            selected_count=int(record["selected_count"]),
            selection_manifest_path=selection_path,
            rendered_manifest_path=rendered_path,
        )

    def _restore_embedding(self, layout: LotLayout) -> EmbeddingBatchResult:
        report_path = layout.reports_dir / "embedding.json"
        payload = self._read_json(report_path)
        dimension = int(payload["dimension"])
        writer = NpyFeatureWriter(dimension)
        records = payload.get("videos")
        if not isinstance(records, list):
            raise ValueError(f"Embedding report is invalid: {report_path}")
        restored: list[EmbeddingVideoResult] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Invalid embedding record: {report_path}")
            feature_files = tuple(Path(str(path)) for path in record.get("feature_files", []))
            image_count = int(record["image_count"])
            if len(feature_files) != image_count:
                raise ValueError(f"Embedding report has incomplete feature list: {report_path}")
            for feature_file in feature_files:
                writer.validate_file(feature_file)
            restored.append(
                EmbeddingVideoResult(
                    video_id=str(record["video_id"]),
                    source_dir=Path(str(record["source_dir"])),
                    output_dir=Path(str(record["output_dir"])),
                    image_count=image_count,
                    embedded_count=int(record["embedded_count"]),
                    skipped_count=int(record["skipped_count"]),
                    dimension=int(record["dimension"]),
                    feature_files=feature_files,
                )
            )
        result = EmbeddingBatchResult(videos=tuple(restored), dimension=dimension)
        if result.embedded_count + result.skipped_count != result.image_count:
            raise ValueError(f"Embedding report counts are inconsistent: {report_path}")
        return result

    def _restore_upload(self, layout: LotLayout) -> UploadResult:
        path = layout.receipts_dir / "upload.json"
        payload = self._read_json(path)
        result = UploadResult(
            dataset_ref=str(payload["dataset_ref"]),
            mode=str(payload["mode"]),
            verified=bool(payload["verified"]),
            command=tuple(str(item) for item in payload.get("command", [])),
            output_tail=str(payload.get("output_tail", "")),
            verified_output_tail=str(payload.get("verified_output_tail", "")),
        )
        if not result.verified:
            raise ValueError(f"Upload receipt is not verified: {path}")
        return result

    def _restore_cleanup(self, layout: LotLayout) -> None:
        result_path = layout.receipts_dir / "cleanup-result.json"
        manager_path = layout.receipts_dir / "cleanup.json"
        if not result_path.is_file() and not manager_path.is_file():
            raise FileNotFoundError(f"Cleanup receipt not found: {result_path}")
        self._read_json(result_path if result_path.is_file() else manager_path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return payload

    def _download(self, request: ArchiveInput, layout: LotLayout, checkpoints: CheckpointStore) -> Path:
        archive_path = layout.archive_dir / request.archive_name
        if archive_path.is_file() and archive_path.stat().st_size > 0:
            try:
                self.archive_validator.validate(archive_path)
                checkpoints.transition(BatchState.DOWNLOADED, payload={"archive_path": str(archive_path)})
                return archive_path
            except (OSError, ValueError):
                # Keep the existing file for inspection; aria2c can resume it.
                pass
        checkpoints.transition(BatchState.DOWNLOADING)
        result = self.downloader.download(request, layout.archive_dir)
        checkpoints.transition(BatchState.DOWNLOADED, payload={"download": result.to_dict()})
        return result.path

    def _validate_archive(
        self,
        archive_path: Path,
        layout: LotLayout,
        checkpoints: CheckpointStore,
    ) -> ArchiveInspection:
        inspection = self.archive_validator.validate(archive_path)
        self._write_json(layout.reports_dir / "archive.json", inspection.to_dict())
        checkpoints.transition(BatchState.ARCHIVE_VALIDATED, payload={"archive": inspection.to_dict()})
        return inspection

    def _extract(
        self,
        inspection: ArchiveInspection,
        request: ArchiveInput,
        layout: LotLayout,
        checkpoints: CheckpointStore,
    ) -> Path:
        if layout.source_root.is_dir():
            try:
                source_root = self._restore_extraction(inspection, layout)
            except (OSError, ValueError) as exc:
                # The normal extractor publishes the root atomically.  This
                # fallback handles an older/externally interrupted extraction
                # by deleting only the lot-owned incomplete source root.
                if not layout.is_owned_path(layout.source_root):
                    raise RuntimeError(f"Refusing to replace source root: {layout.source_root}") from exc
                shutil.rmtree(layout.source_root)
            else:
                checkpoints.transition(BatchState.EXTRACTED, payload={"source_root": str(source_root)})
                return source_root
        source_root = self.archive_extractor.extract(inspection, layout.source_dir, request.lot_id)
        checkpoints.transition(BatchState.EXTRACTED, payload={"source_root": str(source_root)})
        return source_root

    def _discover(
        self,
        layout: LotLayout,
        request: ArchiveInput,
        checkpoints: CheckpointStore,
    ) -> list[VideoAsset]:
        from preprocess.batch.video_discovery import VideoDiscovery

        assets = VideoDiscovery(self.config.archive.video_extensions).discover(layout.source_root, request.lot_id)
        self._write_json(layout.reports_dir / "videos.json", {"videos": [asset.to_dict() for asset in assets]})
        checkpoints.transition(
            BatchState.VIDEOS_DISCOVERED,
            payload={"videos": [asset.to_dict() for asset in assets]},
        )
        return assets

    def _detect_shot_boundaries(
        self,
        assets: Sequence[VideoAsset],
        layout: LotLayout,
        checkpoints: CheckpointStore,
    ) -> tuple[ShotBoundaryArtifact, ...]:
        if self.shot_boundary_detector is None:
            return ()
        output_dir = self.config.processing.scene_segments_dir
        if output_dir is None:
            raise RuntimeError(
                "A scene-boundary output directory is required when automatic shot detection is enabled"
            )

        checkpoints.transition(BatchState.SHOT_BOUNDARIES)
        artifacts = ShotBoundaryPipeline(
            self.shot_boundary_detector,
            output_dir,
            overwrite=self.config.shot_boundary.overwrite,
            progress=self.progress,
        ).run(assets)
        payload = {"videos": [artifact.to_dict() for artifact in artifacts]}
        self._write_json(layout.reports_dir / "shot-boundaries.json", payload)
        checkpoints.transition(BatchState.SHOT_BOUNDARIES_READY, payload={"shot_boundaries": payload})
        return artifacts

    def _process_and_validate(
        self,
        request: ArchiveInput,
        assets: Sequence[VideoAsset],
        layout: LotLayout,
        checkpoints: CheckpointStore,
    ) -> list[ProcessingResult]:
        checkpoints.transition(BatchState.PROCESSING)
        stage_fingerprint = self._stage_fingerprint(request, "process_validate")
        processed: list[ProcessingResult] = []
        for asset in self.progress.iterate(
            assets,
            total=len(assets),
            desc=f"{layout.lot_id}: videos",
            unit="video",
        ):
            video_fingerprint = self._video_checkpoint_fingerprint(stage_fingerprint, asset)
            if checkpoints.video_is_complete("process_validate", asset.video_id, video_fingerprint):
                payload = checkpoints.video_payload("process_validate", asset.video_id)
                if payload is not None:
                    try:
                        processed.append(self._restore_processed_video(asset, layout, payload))
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        checkpoints.invalidate_video(
                            "process_validate",
                            asset.video_id,
                            reason=f"artifact restore failed: {exc}",
                        )
                    else:
                        continue

            checkpoints.start_video(
                "process_validate",
                asset.video_id,
                fingerprint=video_fingerprint,
            )
            metadata = load_video_metadata(self.metadata_provider, asset.video_id)
            result = self.processor.process([asset], layout)[0]
            report = self.video_validator.validate(VideoValidationContext(asset, metadata, result))
            self._write_json(layout.reports_dir / "validation" / f"{asset.video_id}.json", report.to_dict())
            if not report.passed:
                raise RuntimeError(f"Validation failed for {asset.video_id}")
            processed.append(result)
            self._write_json(
                layout.reports_dir / "processing.json",
                {"videos": [item.to_dict() for item in processed]},
            )
            checkpoints.complete_video(
                "process_validate",
                asset.video_id,
                fingerprint=video_fingerprint,
                payload=result.to_dict(),
            )
        self._write_json(
            layout.reports_dir / "processing.json",
            {"videos": [result.to_dict() for result in processed]},
        )
        checkpoints.transition(
            BatchState.PROCESSED,
            payload={"processed": [result.to_dict() for result in processed]},
        )
        checkpoints.transition(BatchState.VALIDATED)
        return processed

    def _embed(
        self,
        assets: Sequence[VideoAsset],
        processed: Sequence[ProcessingResult],
        layout: LotLayout,
        checkpoints: CheckpointStore,
    ) -> EmbeddingBatchResult | None:
        if self.embedding is None:
            return None
        checkpoints.transition(BatchState.EMBEDDING)
        result = self.embedding.embed(layout, assets, processed)
        self._write_json(layout.reports_dir / "embedding.json", result.to_dict())
        checkpoints.transition(BatchState.EMBEDDED, payload={"embedding": result.to_dict()})
        return result

    def _run_upload_stages(
        self,
        request: ArchiveInput,
        assets: Sequence[VideoAsset],
        processed: Sequence[ProcessingResult],
        layout: LotLayout,
        checkpoints: CheckpointStore,
        *,
        reuse_completed: bool = False,
    ) -> UploadResult:
        upload = self._execute_stage(
            checkpoints,
            request,
            "stage_upload",
            action=lambda: self._stage_and_upload(assets, processed, layout, checkpoints),
            restore=lambda: self._restore_upload(layout),
            reuse_completed=reuse_completed,
        )
        self._execute_stage(
            checkpoints,
            request,
            "cleanup",
            action=lambda: self._cleanup(layout, upload, checkpoints),
            restore=lambda: self._restore_cleanup(layout),
            reuse_completed=reuse_completed,
        )
        checkpoints.transition(BatchState.COMPLETED, payload={"finished_at": utc_now()})
        return upload

    def _stage_and_upload(
        self,
        assets: Sequence[VideoAsset],
        processed: Sequence[ProcessingResult],
        layout: LotLayout,
        checkpoints: CheckpointStore,
    ) -> UploadResult | None:
        if not self.config.upload.enabled:
            return None
        if self.stager is None or self.uploader is None:
            raise RuntimeError("Upload is enabled but staging/uploader dependencies are missing")
        if layout.staging_dir.exists():
            if not layout.is_owned_path(layout.staging_dir):
                raise RuntimeError(f"Refusing to replace staging path outside lot: {layout.staging_dir}")
            shutil.rmtree(layout.staging_dir)
        staging = self.stager.stage(layout, assets, processed)
        self._write_json(layout.reports_dir / "staging.json", staging.to_dict())
        checkpoints.transition(BatchState.STAGED, payload=staging.to_dict())
        checkpoints.transition(BatchState.UPLOADING)
        upload = self.uploader.upload_and_verify(staging)
        self._write_json(layout.receipts_dir / "upload.json", upload.to_dict())
        dataset_upload_state = getattr(self, "dataset_upload_state", None)
        if dataset_upload_state is not None:
            dataset_upload_state.record_uploaded(layout.lot_id, upload)
        checkpoints.transition(BatchState.UPLOADED_VERIFIED, payload={"upload": upload.to_dict()})
        return upload

    def _cleanup(self, layout: LotLayout, upload: UploadResult, checkpoints: CheckpointStore) -> None:
        checkpoints.transition(BatchState.CLEANING)
        result = self.cleanup.cleanup(layout, upload)
        self._write_json(layout.receipts_dir / "cleanup-result.json", result.to_dict())

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def build_default_orchestrator(config: BatchConfig) -> BatchOrchestrator:
    """Build the default implementation graph; callers may inject custom parts."""
    device_hint = "auto"
    if config.shot_boundary.enabled and selector_requires_scene_boundaries(config.processing.selector):
        device_hint = config.shot_boundary.device
    elif config.embedding.enabled:
        device_hint = config.embedding.device
    progress = TqdmProgressReporter(
        config.progress,
        disk_root=config.data_root,
        device_hint=device_hint,
    )
    selection_registry: SelectionStrategyRegistry = default_selection_registry()
    selection_strategy = selection_registry.create(config.processing.selector, config.processing)
    processing_registry: ProcessingStrategyRegistry = default_processing_registry()
    processing_strategy = processing_registry.create(
        config.processing.strategy,
        selector_strategy=selection_strategy,
        tools=config.tools,
        processing_config=config.processing,
        progress=progress,
    )
    shot_boundary_detector = None
    if config.shot_boundary.enabled and selector_requires_scene_boundaries(config.processing.selector):
        shot_boundary_detector = default_shot_boundary_registry().create(
            config.shot_boundary.backend,
            config.shot_boundary,
        )
    embedding_strategy = (
        default_pecore_embedding_strategy(
            config.embedding,
            input_profile_id=config.processing.profile_id,
            progress=progress,
        )
        if config.embedding.enabled
        else None
    )
    metadata_provider = JsonMetadataProvider(config.metadata_root)
    dataset_upload_state = None
    dataset_upload_lock = None
    stager = None
    if config.upload.enabled:
        if config.upload.staging_scope == "dataset":
            dataset_staging_dir = (
                config.upload.dataset_staging_dir
                or config.data_root / "kaggle-dataset-staging"
            )
            dataset_upload_state = DatasetUploadStateStore(dataset_state_path(dataset_staging_dir))
            dataset_upload_lock = dataset_upload_lock_path(dataset_staging_dir)
            stager = CumulativeKaggleStagingStrategy(
                metadata_provider,
                config.upload,
                staging_dir=dataset_staging_dir,
                state_store=dataset_upload_state,
                scene_segments_dir=config.processing.scene_segments_dir,
            )
        else:
            stager = KaggleStagingStrategy(
                metadata_provider,
                config.upload,
                scene_segments_dir=config.processing.scene_segments_dir,
            )
    uploader = KaggleCliUploader(config.tools.kaggle, config.upload) if config.upload.enabled else None
    return BatchOrchestrator(
        config,
        downloader=Aria2ArchiveDownloader(
            config.tools.aria2c,
            config.download,
            show_progress=config.progress.enabled,
        ),
        archive_validator=ZipArchiveValidator(config.archive),
        archive_extractor=ZipArchiveExtractor(),
        metadata_provider=metadata_provider,
        processor=BatchProcessor(processing_strategy),
        video_validator=default_video_validation_pipeline(
            tools=config.tools,
            processing_config=config.processing,
        ),
        stager=stager,
        uploader=uploader,
        cleanup=CleanupManager(
            config.cleanup,
            rendered_profile_id=config.processing.profile_id,
            preserve_staging=config.upload.staging_scope == "dataset",
        ),
        shot_boundary_detector=shot_boundary_detector,
        embedding=embedding_strategy,
        progress=progress,
        dataset_upload_state=dataset_upload_state,
        dataset_upload_lock_path=dataset_upload_lock,
    )
