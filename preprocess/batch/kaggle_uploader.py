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
from preprocess.batch.layout import LotLayout
from preprocess.batch.metadata import MetadataProvider
from preprocess.batch.models import ProcessingResult, StagingResult, UploadResult, VideoAsset


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

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{layout.staging_dir.name}.", dir=layout.root))
        try:
            self._copy_file(metadata_template, staging_dir / "dataset-metadata.json")

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

            files = tuple(sorted(path for path in staging_dir.rglob("*") if path.is_file()))
            if layout.staging_dir.exists():
                if any(layout.staging_dir.iterdir()):
                    raise FileExistsError(f"Kaggle staging directory is not empty: {layout.staging_dir}")
                layout.staging_dir.rmdir()
            os.replace(staging_dir, layout.staging_dir)
            final_files = tuple(
                layout.staging_dir / path.relative_to(staging_dir) for path in files
            )
            return StagingResult(staging_dir=layout.staging_dir, files=final_files)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

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
        command = self._upload_command(staging.staging_dir)
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            raise RuntimeError(f"Kaggle upload failed (exit {completed.returncode}): {output[-3000:]}")
        verified, verification_output = self._verify()
        if not verified:
            raise RuntimeError(f"Kaggle upload was not verified: {verification_output[-3000:]}")
        return UploadResult(
            dataset_ref=self.config.dataset_ref or "created-from-metadata",
            mode=self.config.mode,
            verified=True,
            command=tuple(command),
            output_tail=output[-3000:],
            verified_output_tail=verification_output[-3000:],
        )

    def _upload_command(self, staging_dir: Path) -> list[str]:
        if self.config.mode == "create":
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
        if not self.config.dataset_ref:
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

    def _verify(self) -> tuple[bool, str]:
        if not self.config.dataset_ref:
            return False, "dataset_ref is required for remote verification"
        command = [
            self.executable,
            "datasets",
            "status",
            self.config.dataset_ref,
        ]
        deadline = time.monotonic() + self.config.verify_timeout_seconds
        output = ""
        while time.monotonic() <= deadline:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            output = (completed.stdout or "") + (completed.stderr or "")
            if completed.returncode == 0:
                return True, output
            time.sleep(self.config.verify_poll_seconds)
        return False, output
