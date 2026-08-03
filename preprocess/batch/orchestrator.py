"""Dependency-injected orchestration for one or more archive-derived lots."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from preprocess.batch.archive_extractor import ArchiveExtractor, ZipArchiveExtractor
from preprocess.batch.archive_validator import ZipArchiveValidator
from preprocess.batch.checkpoints import CheckpointStore
from preprocess.batch.config import BatchConfig, selector_requires_scene_boundaries
from preprocess.batch.cleanup import CleanupManager
from preprocess.batch.downloader import Aria2ArchiveDownloader, ArchiveDownloader
from preprocess.batch.embedding import (
    BatchEmbeddingStrategy,
    default_pecore_embedding_strategy,
)
from preprocess.batch.layout import LotLayout
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
)
from preprocess.pecore.embedding import EmbeddingBatchResult
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

    def run_all(self, requests: Sequence[ArchiveInput]) -> list[LotRunResult]:
        PreflightChecker(self.config).run()
        results: list[LotRunResult] = []
        for request in self.progress.iterate(
            requests,
            total=len(requests),
            desc="lots",
            unit="lot",
        ):
            results.append(self.run_lot(request))
        return results

    def run_lot(self, request: ArchiveInput) -> LotRunResult:
        layout = LotLayout(self.config.data_root, request.lot_id)
        layout.create_runtime_dirs()
        checkpoints = CheckpointStore(layout.state_path)
        self._initialize_state(checkpoints, request)
        try:
            archive_path = self._download(request, layout, checkpoints)
            inspection = self._validate_archive(archive_path, layout, checkpoints)
            self._extract(inspection, request, layout, checkpoints)
            assets = self._discover(layout, request, checkpoints)
            self._detect_shot_boundaries(assets, layout, checkpoints)
            processed = self._process_and_validate(assets, layout, checkpoints)
            embedding = self._embed(assets, processed, layout, checkpoints)
            upload = self._stage_and_upload(assets, processed, layout, checkpoints)
            if upload is not None:
                self._cleanup(layout, upload, checkpoints)
            checkpoints.transition(BatchState.COMPLETED, payload={"finished_at": utc_now()})
            return LotRunResult(request.lot_id, tuple(assets), tuple(processed), upload, embedding)
        except Exception as exc:
            checkpoints.transition(
                BatchState.FAILED,
                payload={"error": f"{type(exc).__name__}: {exc}", "failed_at": utc_now()},
            )
            raise

    def _initialize_state(self, checkpoints: CheckpointStore, request: ArchiveInput) -> None:
        state = checkpoints.load()
        if state.get("state") == BatchState.COMPLETED.value:
            raise RuntimeError(f"Lot is already completed: {request.lot_id}")
        if not state.get("initialized"):
            checkpoints.write(
                {
                    "state": BatchState.NEW.value,
                    "initialized": True,
                    "lot_id": request.lot_id,
                    "request": request.to_dict(),
                    "config": self.config.to_dict(),
                    "created_at": utc_now(),
                    "events": [],
                }
            )

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
            checkpoints.transition(BatchState.EXTRACTED, payload={"source_root": str(layout.source_root)})
            return layout.source_root
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
        assets: Sequence[VideoAsset],
        layout: LotLayout,
        checkpoints: CheckpointStore,
    ) -> list[ProcessingResult]:
        checkpoints.transition(BatchState.PROCESSING)
        processed: list[ProcessingResult] = []
        for asset in self.progress.iterate(
            assets,
            total=len(assets),
            desc=f"{layout.lot_id}: videos",
            unit="video",
        ):
            metadata = load_video_metadata(self.metadata_provider, asset.video_id)
            result = self.processor.process([asset], layout)[0]
            report = self.video_validator.validate(VideoValidationContext(asset, metadata, result))
            self._write_json(layout.reports_dir / "validation" / f"{asset.video_id}.json", report.to_dict())
            if not report.passed:
                raise RuntimeError(f"Validation failed for {asset.video_id}")
            processed.append(result)
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
        staging = self.stager.stage(layout, assets, processed)
        self._write_json(layout.reports_dir / "staging.json", staging.to_dict())
        checkpoints.transition(BatchState.STAGED, payload=staging.to_dict())
        checkpoints.transition(BatchState.UPLOADING)
        upload = self.uploader.upload_and_verify(staging)
        self._write_json(layout.receipts_dir / "upload.json", upload.to_dict())
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
    progress = TqdmProgressReporter(config.progress)
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
    stager = KaggleStagingStrategy(metadata_provider, config.upload) if config.upload.enabled else None
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
        ),
        shot_boundary_detector=shot_boundary_detector,
        embedding=embedding_strategy,
        progress=progress,
    )
