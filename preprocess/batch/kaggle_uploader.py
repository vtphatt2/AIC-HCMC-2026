"""Kaggle staging and uploader adapters.

The stager deliberately excludes raw archives and source videos.  Uploading is
behind an interface so a local fake, S3 adapter, or another dataset service can
be used without changing the processing pipeline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from preprocess.batch.config import UploadConfig
from preprocess.batch.dataset_state import DatasetUploadStateStore
from preprocess.batch.layout import LotLayout
from preprocess.batch.metadata import MetadataProvider
from preprocess.batch.models import (
    DatasetTarget,
    ProcessingResult,
    StagingResult,
    UploadResult,
    VideoAsset,
)


class StagingStrategy(ABC):
    """Build an explicit, allowlisted local upload payload."""

    @abstractmethod
    def stage(
        self,
        layout: LotLayout,
        assets: Sequence[VideoAsset],
        results: Sequence[ProcessingResult],
    ) -> StagingResult:
        raise NotImplementedError


class KaggleStagingStrategy(StagingStrategy):
    """Stage allowlisted artifacts; never stage raw archive, source, or video."""

    def __init__(
        self,
        metadata_provider: MetadataProvider,
        config: UploadConfig,
        scene_segments_dir: Path | None = None,
    ) -> None:
        self.metadata_provider = metadata_provider
        self.config = config
        self.scene_segments_dir = scene_segments_dir

    def stage(
        self,
        layout: LotLayout,
        assets: Sequence[VideoAsset],
        results: Sequence[ProcessingResult],
    ) -> StagingResult:
        if layout.staging_dir.exists() and any(layout.staging_dir.iterdir()):
            raise FileExistsError(f"Kaggle staging directory is not empty: {layout.staging_dir}")
        target = self.config.target_for_lot(layout.lot_id)
        staging_dir = self._build_payload(layout, assets, results, parent=layout.root)
        try:
            files = tuple(sorted(path for path in staging_dir.rglob("*") if path.is_file()))
            if layout.staging_dir.exists():
                if any(layout.staging_dir.iterdir()):
                    raise FileExistsError(f"Kaggle staging directory is not empty: {layout.staging_dir}")
                layout.staging_dir.rmdir()
            os.replace(staging_dir, layout.staging_dir)
            final_files = tuple(
                layout.staging_dir / path.relative_to(staging_dir) for path in files
            )
            return StagingResult(
                staging_dir=layout.staging_dir,
                files=final_files,
                target=target,
            )
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _build_payload(
        self,
        layout: LotLayout,
        assets: Sequence[VideoAsset],
        results: Sequence[ProcessingResult],
        *,
        parent: Path,
    ) -> Path:
        metadata_template = self.config.metadata_template
        if metadata_template is None or not metadata_template.is_file():
            raise FileNotFoundError(
                "upload.metadata_template is required and must point to dataset-metadata.json"
            )

        scene_segment_paths: dict[str, Path] = {}
        if self.config.include_scene_segments:
            if self.scene_segments_dir is None:
                raise ValueError(
                    "upload.include_scene_segments=true requires "
                    "processing.scene_segments_dir"
                )
            for asset in assets:
                scene_segments_path = self.scene_segments_dir / f"{asset.video_id}.json"
                if not scene_segments_path.is_file():
                    raise FileNotFoundError(
                        "Scene-segment manifest not found for "
                        f"{asset.video_id}: {scene_segments_path}"
                    )
                scene_segment_paths[asset.video_id] = scene_segments_path

        parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".kaggle-staging.", dir=parent))
        try:
            target = self.config.target_for_lot(layout.lot_id)
            self._write_dataset_metadata(
                metadata_template,
                staging_dir / "dataset-metadata.json",
                target,
            )

            for asset, result in zip(assets, results, strict=True):
                source_keyframes = result.rendered_manifest_path.parent
                destination_keyframes = staging_dir / "keyframes" / asset.video_id
                self._copy_tree(source_keyframes, destination_keyframes, skip_names={"manifest.json"})
                self._copy_file(
                    self.metadata_provider.path_for(asset.video_id),
                    staging_dir / "metadata" / f"{asset.video_id}.json",
                )
                self._copy_tree(
                    result.selection_manifest_path.parent,
                    staging_dir / "manifests" / "selection",
                    include_names={result.selection_manifest_path.name},
                )
                rendered_destination = (
                    staging_dir / "manifests" / "rendered" / f"{asset.video_id}.json"
                )
                self._copy_file(result.rendered_manifest_path, rendered_destination)
                validation_path = layout.reports_dir / "validation" / f"{asset.video_id}.json"
                if validation_path.is_file():
                    self._copy_file(
                        validation_path,
                        staging_dir / "manifests" / "validation" / validation_path.name,
                    )

                if self.config.include_scene_segments:
                    scene_segments_path = scene_segment_paths[asset.video_id]
                    self._copy_file(
                        scene_segments_path,
                        staging_dir / "scene-segments" / scene_segments_path.name,
                    )

                if self.config.include_features:
                    feature_dir = layout.dataset_dir / "PECore-features" / asset.video_id
                    if feature_dir.is_dir():
                        self._copy_tree(feature_dir, staging_dir / "PECore-features" / asset.video_id)

            if self.config.include_transcripts:
                transcript_dir = layout.dataset_dir / "transcripts"
                if transcript_dir.is_dir():
                    self._copy_tree(transcript_dir, staging_dir / "transcripts")
            if self.config.include_transcript_index:
                index_dir = layout.dataset_dir / "keyframe_transcript_index"
                if index_dir.is_dir():
                    self._copy_tree(index_dir, staging_dir / "keyframe_transcript_index")
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        return staging_dir

    @staticmethod
    def _write_dataset_metadata(
        source: Path,
        destination: Path,
        target: DatasetTarget,
    ) -> None:
        """Copy the metadata template while binding it to this lot's dataset."""
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dataset metadata must be valid JSON: {source}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Dataset metadata must be a JSON object: {source}")
        metadata = dict(payload)
        metadata["id"] = target.dataset_ref
        metadata.pop("id_no", None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _copy_file(source: Path, destination: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Staging source file not found: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def _copy_tree(
        self,
        source: Path,
        destination: Path,
        *,
        skip_names: set[str] | None = None,
        include_names: set[str] | None = None,
    ) -> list[Path]:
        if not source.is_dir():
            raise FileNotFoundError(f"Staging source directory not found: {source}")
        copied: list[Path] = []
        skip_names = skip_names or set()
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name in skip_names:
                continue
            if include_names is not None and path.name not in include_names:
                continue
            relative = path.relative_to(source)
            target = destination / relative
            self._copy_file(path, target)
            copied.append(target)
        return copied


class CumulativeKaggleStagingStrategy(KaggleStagingStrategy):
    """Merge each lot into one persistent dataset-level upload snapshot."""

    def __init__(
        self,
        metadata_provider: MetadataProvider,
        config: UploadConfig,
        *,
        staging_dir: Path,
        state_store: DatasetUploadStateStore,
        scene_segments_dir: Path | None = None,
    ) -> None:
        super().__init__(
            metadata_provider,
            config,
            scene_segments_dir=scene_segments_dir,
        )
        self.staging_dir = staging_dir
        self.state_store = state_store

    def stage(
        self,
        layout: LotLayout,
        assets: Sequence[VideoAsset],
        results: Sequence[ProcessingResult],
    ) -> StagingResult:
        if self.staging_dir.resolve() == layout.data_root.resolve():
            raise ValueError("Cumulative dataset staging must not be the data root itself")
        if self.staging_dir.exists() and not self.staging_dir.is_dir():
            raise FileExistsError(f"Cumulative staging path is not a directory: {self.staging_dir}")

        self.state_store.initialize(
            dataset_ref=self.config.dataset_ref or "created-from-metadata",
            staging_dir=self.staging_dir,
        )
        self.state_store.ensure_staging_available(self.staging_dir)
        payload_dir = self._build_payload(
            layout,
            assets,
            results,
            parent=self.staging_dir.parent,
        )
        try:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            for asset in assets:
                self._remove_video_payload(asset.video_id)
            self._merge_payload(payload_dir)
            files = tuple(sorted(path for path in self.staging_dir.rglob("*") if path.is_file()))
            relative_files = tuple(
                path.relative_to(self.staging_dir).as_posix() for path in files
            )
            self.state_store.record_staged(
                layout.lot_id,
                video_ids=[asset.video_id for asset in assets],
                files=relative_files,
            )
            return StagingResult(
                staging_dir=self.staging_dir,
                files=files,
                target=self.config.target_for_lot(layout.lot_id),
            )
        finally:
            shutil.rmtree(payload_dir, ignore_errors=True)

    def _remove_video_payload(self, video_id: str) -> None:
        targets = (
            self.staging_dir / "keyframes" / video_id,
            self.staging_dir / "metadata" / f"{video_id}.json",
            self.staging_dir / "manifests" / "selection" / f"{video_id}.json",
            self.staging_dir / "manifests" / "rendered" / f"{video_id}.json",
            self.staging_dir / "manifests" / "validation" / f"{video_id}.json",
            self.staging_dir / "scene-segments" / f"{video_id}.json",
            self.staging_dir / "PECore-features" / video_id,
        )
        for target in targets:
            self._remove_owned_path(target)

    def _remove_owned_path(self, path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        try:
            path.resolve().relative_to(self.staging_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Refusing to modify path outside cumulative staging: {path}") from exc
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _merge_payload(self, payload_dir: Path) -> None:
        for source in sorted(path for path in payload_dir.rglob("*") if path.is_file()):
            relative = source.relative_to(payload_dir)
            destination = self.staging_dir / relative
            self._replace_file(source, destination)

    @staticmethod
    def _replace_file(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


class DatasetUploader(ABC):
    """Replaceable remote upload boundary."""

    @abstractmethod
    def upload_and_verify(self, staging: StagingResult) -> UploadResult:
        raise NotImplementedError


class KaggleCliUploader(DatasetUploader):
    """Use the official Kaggle CLI for create/version and status verification."""

    def __init__(self, executable: str, config: UploadConfig) -> None:
        self.executable = executable
        self.config = config

    def upload_and_verify(self, staging: StagingResult) -> UploadResult:
        if not self.config.enabled:
            raise RuntimeError("Kaggle upload is disabled in configuration")
        target = staging.target
        if target is None:
            if not self.config.dataset_ref:
                raise ValueError(
                    "Staging result has no dataset target and upload.dataset_ref is not configured"
                )
            target = DatasetTarget(
                dataset_ref=self.config.dataset_ref,
                mode=self.config.mode,
            )
        mode = self._resolve_mode(target)
        command = self._upload_command(
            staging.staging_dir,
            dataset_ref=target.dataset_ref,
            mode=mode,
        )
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            raise RuntimeError(f"Kaggle upload failed (exit {completed.returncode}): {output[-3000:]}")
        verified, verification_output = self._verify(target.dataset_ref)
        if not verified:
            raise RuntimeError(f"Kaggle upload was not verified: {verification_output[-3000:]}")
        return UploadResult(
            dataset_ref=target.dataset_ref,
            mode=mode,
            verified=True,
            command=tuple(command),
            output_tail=output[-3000:],
            verified_output_tail=verification_output[-3000:],
        )

    def _resolve_mode(self, target: DatasetTarget) -> str:
        if target.mode in {"create", "version"}:
            return target.mode
        if target.mode != "auto":
            raise ValueError(f"Unsupported upload mode: {target.mode!r}")
        completed = subprocess.run(
            self._status_command(target.dataset_ref),
            text=True,
            capture_output=True,
            check=False,
        )
        return "version" if completed.returncode == 0 else "create"

    def _upload_command(
        self,
        staging_dir: Path,
        *,
        dataset_ref: str | None = None,
        mode: str | None = None,
    ) -> list[str]:
        effective_mode = mode or self.config.mode
        effective_ref = dataset_ref or self.config.dataset_ref
        if effective_mode == "create":
            command = [
                self.executable,
                "datasets",
                "create",
                "-p",
                str(staging_dir),
                "--dir-mode",
                self.config.dir_mode,
            ]
            command.append("--public" if self.config.public else "--private")
            return command
        if effective_mode != "version":
            raise ValueError(
                "_upload_command requires resolved mode 'create' or 'version'"
            )
        if not effective_ref:
            raise ValueError("dataset_ref is required for version uploads")
        return [
            self.executable,
            "datasets",
            "version",
            "-p",
            str(staging_dir),
            "-m",
            self.config.version_message,
            "--dir-mode",
            self.config.dir_mode,
        ]

    def _status_command(self, dataset_ref: str) -> list[str]:
        return [
            self.executable,
            "datasets",
            "status",
            dataset_ref,
        ]

    def _verify(self, dataset_ref: str | None = None) -> tuple[bool, str]:
        effective_ref = dataset_ref or self.config.dataset_ref
        if not effective_ref:
            return False, "dataset_ref is required for remote verification"
        command = self._status_command(effective_ref)
        deadline = time.monotonic() + self.config.verify_timeout_seconds
        output = ""
        while time.monotonic() <= deadline:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            output = (completed.stdout or "") + (completed.stderr or "")
            if completed.returncode == 0:
                return True, output
            time.sleep(self.config.verify_poll_seconds)
        return False, output
