"""Kaggle staging and uploader adapters.

The stager deliberately excludes raw archives and source videos.  Uploading is
behind an interface so a local fake, S3 adapter, or another dataset service can
be used without changing the processing pipeline.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    utc_now,
)
from preprocess.batch.provenance import (
    PROVENANCE_FILE_NAME,
    atomic_json_write,
    digest_directory,
    file_fingerprint_matches,
    runtime_provenance,
)
from preprocess.progress import ProgressReporter


_KAGGLE_TRANSFER_PROGRESS = re.compile(
    r"(?P<percent>\d{1,3})%\|[^\r\n]*?\|\s*"
    r"(?P<current>\d+(?:\.\d+)?)(?P<current_unit>[kKMGTPE]?)/"
    r"(?P<total>\d+(?:\.\d+)?)(?P<total_unit>[kKMGTPE]?)"
)


def _scaled_transfer_bytes(value: str, unit: str) -> int:
    powers = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}
    return round(float(value) * (1000 ** powers[unit.lower()]))


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
        provenance_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.metadata_provider = metadata_provider
        self.config = config
        self.scene_segments_dir = scene_segments_dir
        self.provenance_context = dict(provenance_context or {})

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
            payload_digest = self._write_provenance(staging_dir, layout, target)
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
                payload_digest=payload_digest,
                provenance_path=layout.staging_dir / PROVENANCE_FILE_NAME,
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
                "upload.metadata_template must point to an existing Kaggle metadata "
                "template JSON; staging will name its copy dataset-metadata.json"
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
                if result.asset.video_id != asset.video_id:
                    raise ValueError(
                        f"Processing result does not match asset: {asset.video_id} != "
                        f"{result.asset.video_id}"
                    )
                destination_keyframes = staging_dir / "keyframes" / asset.video_id
                self._copy_keyframes_from_manifest(
                    result.rendered_manifest_path,
                    destination_keyframes,
                )
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
                    feature_root = layout.dataset_dir / self._features_dir_name(layout)
                    if not feature_root.is_dir():
                        if self.config.missing_artifact_policy == "error":
                            raise FileNotFoundError(
                                f"Feature directory not found for {asset.video_id}: {feature_root}"
                            )
                        feature_files = ()
                    else:
                        feature_files = self._feature_files_for_video(layout, asset.video_id)
                        if self.config.missing_artifact_policy == "error" and not feature_files:
                            raise FileNotFoundError(
                                f"No feature files found for {asset.video_id} while staging"
                            )
                    for feature_file in feature_files:
                        relative = feature_file.relative_to(
                            layout.dataset_dir / self._features_dir_name(layout)
                        )
                        self._copy_file(
                            feature_file,
                            staging_dir / "PECore-features" / relative,
                        )

            if self.config.include_transcripts:
                transcript_dir = layout.dataset_dir / "transcripts"
                if transcript_dir.is_dir():
                    copied = self._copy_tree_for_video_ids(
                        transcript_dir,
                        staging_dir / "transcripts",
                        {asset.video_id for asset in assets},
                    )
                    if self.config.missing_artifact_policy == "error":
                        missing = [
                            asset.video_id
                            for asset in assets
                            if not any(
                                self._belongs_to_video(path, asset.video_id)
                                for path in copied
                            )
                        ]
                        if missing:
                            raise FileNotFoundError(
                                "Transcript artifact is missing for: " + ", ".join(missing)
                            )
                elif self.config.missing_artifact_policy == "error":
                    raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
            if self.config.include_transcript_index:
                index_dir = layout.dataset_dir / "keyframe_transcript_index"
                if index_dir.is_dir():
                    copied = self._copy_tree_for_video_ids(
                        index_dir,
                        staging_dir / "keyframe_transcript_index",
                        {asset.video_id for asset in assets},
                    )
                    if self.config.missing_artifact_policy == "error":
                        missing = [
                            asset.video_id
                            for asset in assets
                            if not any(
                                self._belongs_to_video(path, asset.video_id)
                                for path in copied
                            )
                        ]
                        if missing:
                            raise FileNotFoundError(
                                "Transcript index artifact is missing for: " + ", ".join(missing)
                            )
                elif self.config.missing_artifact_policy == "error":
                    raise FileNotFoundError(f"Transcript index directory not found: {index_dir}")
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        return staging_dir

    @staticmethod
    def _features_dir_name(layout: LotLayout) -> str:
        del layout
        return "PECore-features"

    def _feature_files_for_video(self, layout: LotLayout, video_id: str) -> tuple[Path, ...]:
        """Read the embedding report and return only its allowlisted files."""
        feature_root = layout.dataset_dir / self._features_dir_name(layout)
        video_root = feature_root / video_id
        if not video_root.is_dir():
            return ()
        report_path = layout.reports_dir / "embedding.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"Embedding report is required to stage features: {report_path}"
            )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        records = payload.get("videos") if isinstance(payload, Mapping) else None
        record = next(
            (item for item in records or [] if isinstance(item, Mapping) and str(item.get("video_id")) == video_id),
            None,
        )
        if record is None:
            raise ValueError(f"Embedding report does not contain {video_id}: {report_path}")
        files: list[Path] = []
        for raw_path in record.get("feature_files", []):
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = layout.root / path
            try:
                path.resolve().relative_to(video_root.resolve())
            except ValueError as exc:
                raise ValueError(f"Feature path is outside its video directory: {path}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"Feature file is missing: {path}")
            if path.suffix.lower() != ".npy":
                raise ValueError(f"Embedding report points to a non-NPY file: {path}")
            files.append(path)
        video_provenance = video_root / "provenance.json"
        if video_provenance.is_file():
            files.append(video_provenance)
        else:
            legacy_sidecars = sorted(video_root.glob("*.npy.meta.json"))
            if len(legacy_sidecars) != len(files):
                raise FileNotFoundError(
                    f"Feature provenance is incomplete for {video_id}: {video_root}"
                )
            files.extend(legacy_sidecars)
        if len(set(files)) != len(files):
            raise ValueError(f"Embedding report contains duplicate feature files for {video_id}")
        return tuple(sorted(files))

    def _copy_keyframes_from_manifest(self, manifest_path: Path, destination: Path) -> None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Rendered manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = payload.get("frames") if isinstance(payload, Mapping) else None
        if not isinstance(frames, list) or not frames:
            # Keep a narrow migration path for pre-manifest fixtures: a
            # directory containing exactly one image is unambiguous. Any
            # multi-frame legacy directory must be reprocessed first.
            legacy_images = sorted(
                path
                for path in manifest_path.parent.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if len(legacy_images) != 1:
                raise ValueError(f"Rendered manifest has no allowlisted frames: {manifest_path}")
            self._copy_file(legacy_images[0], destination / legacy_images[0].name)
            return
        source_root = manifest_path.parent.resolve()
        copied_names: set[str] = set()
        for frame in frames:
            if not isinstance(frame, Mapping) or not frame.get("path"):
                raise ValueError(f"Rendered manifest contains an invalid frame: {manifest_path}")
            path = Path(str(frame["path"]))
            if not path.is_absolute():
                path = manifest_path.parent / path
            try:
                path.resolve().relative_to(source_root)
            except ValueError as exc:
                raise ValueError(f"Rendered frame is outside its keyframe directory: {path}") from exc
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Rendered manifest points to a non-image file: {path}")
            if path.name in copied_names:
                raise ValueError(f"Rendered manifest contains duplicate frame path: {path.name}")
            copied_names.add(path.name)
            self._copy_file(path, destination / path.name)

    def _write_provenance(
        self,
        staging_dir: Path,
        layout: LotLayout,
        target: DatasetTarget,
    ) -> str:
        digest, records = digest_directory(
            staging_dir,
            exclude_names={PROVENANCE_FILE_NAME},
        )
        payload: dict[str, Any] = {
            **self.provenance_context,
            "schema_version": 1,
            "lot_id": layout.lot_id,
            "dataset_ref": target.dataset_ref,
            "payload_digest": digest,
            "files": records,
            "runtime": runtime_provenance(
                root=Path(__file__).resolve().parents[2],
                tools=(
                    self.provenance_context.get("config", {}).get("tools")
                    if isinstance(self.provenance_context.get("config"), Mapping)
                    else None
                ),
            ),
        }
        atomic_json_write(staging_dir / PROVENANCE_FILE_NAME, payload)
        return digest

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

    def _copy_tree_for_video_ids(
        self,
        source: Path,
        destination: Path,
        video_ids: set[str],
    ) -> list[Path]:
        if not source.is_dir():
            raise FileNotFoundError(f"Staging source directory not found: {source}")
        copied: list[Path] = []
        for path in sorted(source.rglob("*")):
            if not path.is_file() or not any(self._belongs_to_video(path, video_id) for video_id in video_ids):
                continue
            target = destination / path.relative_to(source)
            self._copy_file(path, target)
            copied.append(target)
        return copied

    @staticmethod
    def _belongs_to_video(path: Path, video_id: str) -> bool:
        return path.stem == video_id or path.stem.startswith(f"{video_id}_")


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
        provenance_context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            metadata_provider,
            config,
            scene_segments_dir=scene_segments_dir,
            provenance_context=provenance_context,
        )
        self.staging_dir = staging_dir
        self.state_store = state_store

    @property
    def transaction_path(self) -> Path:
        return self.staging_dir.parent / "kaggle-dataset-transaction.json"

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
        if self.staging_dir.is_symlink():
            raise RuntimeError(f"Cumulative staging must not be a symlink: {self.staging_dir}")

        self._recover_transaction()
        target = self.config.target_for_lot(layout.lot_id)
        self.state_store.initialize(
            dataset_ref=target.dataset_ref,
            staging_dir=self.staging_dir,
        )
        self.state_store.ensure_staging_available(self.staging_dir)
        payload_dir = self._build_payload(
            layout,
            assets,
            results,
            parent=self.staging_dir.parent,
        )
        parent = self.staging_dir.parent
        candidate_dir = Path(
            tempfile.mkdtemp(prefix=f".{self.staging_dir.name}.candidate.", dir=parent)
        )
        shutil.rmtree(candidate_dir)
        backup_dir = Path(
            tempfile.mkdtemp(prefix=f".{self.staging_dir.name}.backup.", dir=parent)
        )
        shutil.rmtree(backup_dir)
        transaction_status = "building"
        try:
            atomic_json_write(
                self.transaction_path,
                {
                    "schema_version": 1,
                    "status": transaction_status,
                    "staging_dir": str(self.staging_dir),
                    "candidate_dir": str(candidate_dir),
                    "backup_dir": str(backup_dir),
                    "lot_id": layout.lot_id,
                    "started_at": utc_now(),
                },
            )
            if self.staging_dir.is_dir():
                shutil.copytree(
                    self.staging_dir,
                    candidate_dir,
                    copy_function=self._link_or_copy,
                )
            else:
                candidate_dir.mkdir(parents=True, exist_ok=True)
            for asset in assets:
                self._remove_video_payload(asset.video_id, root=candidate_dir)
            self._merge_payload(payload_dir, staging_root=candidate_dir)
            payload_digest = self._write_provenance(candidate_dir, layout, target)
            atomic_json_write(
                self.transaction_path,
                {
                    "schema_version": 1,
                    "status": "ready",
                    "staging_dir": str(self.staging_dir),
                    "candidate_dir": str(candidate_dir),
                    "backup_dir": str(backup_dir),
                    "lot_id": layout.lot_id,
                    "payload_digest": payload_digest,
                    "ready_at": utc_now(),
                },
            )
            transaction_status = "ready"
            self._commit_transaction(candidate_dir, backup_dir)
            files = tuple(sorted(path for path in self.staging_dir.rglob("*") if path.is_file()))
            relative_files = tuple(
                path.relative_to(self.staging_dir).as_posix() for path in files
            )
            self.state_store.record_staged(
                layout.lot_id,
                video_ids=[asset.video_id for asset in assets],
                files=relative_files,
                payload_digest=payload_digest,
            )
            return StagingResult(
                staging_dir=self.staging_dir,
                files=files,
                target=target,
                payload_digest=payload_digest,
                provenance_path=self.staging_dir / PROVENANCE_FILE_NAME,
            )
        except Exception:
            if transaction_status == "building":
                shutil.rmtree(candidate_dir, ignore_errors=True)
                backup_dir_missing = not backup_dir.exists()
                if backup_dir_missing:
                    self.transaction_path.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(payload_dir, ignore_errors=True)

    def _recover_transaction(self) -> None:
        """Finish or roll back a directory swap interrupted by SSH/process loss."""
        if not self.transaction_path.is_file():
            return
        try:
            payload = json.loads(self.transaction_path.read_text(encoding="utf-8"))
            candidate = Path(str(payload["candidate_dir"]))
            backup = Path(str(payload["backup_dir"]))
            status = str(payload.get("status", "building"))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cumulative staging transaction marker is invalid: {self.transaction_path}"
            ) from exc
        parent = self.staging_dir.parent.resolve()
        for path in (candidate, backup):
            try:
                if path.resolve().parent != parent:
                    raise ValueError(path)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Cumulative staging transaction points outside its parent: {path}"
                ) from exc

        if status != "ready":
            if not self.staging_dir.exists() and backup.exists():
                os.replace(backup, self.staging_dir)
            shutil.rmtree(candidate, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
            self.transaction_path.unlink(missing_ok=True)
            return

        if self.staging_dir.exists() and backup.exists():
            # The candidate was already published; only backup cleanup remained.
            shutil.rmtree(backup, ignore_errors=True)
        elif not self.staging_dir.exists() and candidate.exists():
            os.replace(candidate, self.staging_dir)
            shutil.rmtree(backup, ignore_errors=True)
        elif not self.staging_dir.exists() and backup.exists():
            # The process moved the old directory but did not publish candidate.
            os.replace(backup, self.staging_dir)
            shutil.rmtree(candidate, ignore_errors=True)
        else:
            shutil.rmtree(candidate, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
        self.transaction_path.unlink(missing_ok=True)

    def _commit_transaction(self, candidate: Path, backup: Path) -> None:
        if self.staging_dir.exists():
            os.replace(self.staging_dir, backup)
        os.replace(candidate, self.staging_dir)
        shutil.rmtree(backup, ignore_errors=True)
        self.transaction_path.unlink(missing_ok=True)

    def _remove_video_payload(self, video_id: str, *, root: Path | None = None) -> None:
        staging_root = root or self.staging_dir
        targets = (
            staging_root / "keyframes" / video_id,
            staging_root / "metadata" / f"{video_id}.json",
            staging_root / "manifests" / "selection" / f"{video_id}.json",
            staging_root / "manifests" / "rendered" / f"{video_id}.json",
            staging_root / "manifests" / "validation" / f"{video_id}.json",
            staging_root / "scene-segments" / f"{video_id}.json",
            staging_root / "PECore-features" / video_id,
        )
        for target in targets:
            self._remove_owned_path(target, root=staging_root)
        for directory_name in ("transcripts", "keyframe_transcript_index"):
            directory = staging_root / directory_name
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and self._belongs_to_video(path, video_id):
                    self._remove_owned_path(path, root=staging_root)

    def _remove_owned_path(self, path: Path, *, root: Path | None = None) -> None:
        root = root or self.staging_dir
        if not path.exists() and not path.is_symlink():
            return
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Refusing to modify path outside cumulative staging: {path}") from exc
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _merge_payload(self, payload_dir: Path, *, staging_root: Path | None = None) -> None:
        staging_root = staging_root or self.staging_dir
        for source in sorted(path for path in payload_dir.rglob("*") if path.is_file()):
            relative = source.relative_to(payload_dir)
            destination = staging_root / relative
            self._replace_file(source, destination)

    @staticmethod
    def _link_or_copy(source: str, destination: str) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    @staticmethod
    def _belongs_to_video(path: Path, video_id: str) -> bool:
        return path.stem == video_id or path.stem.startswith(f"{video_id}_")

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

    def verify_existing(
        self,
        upload: UploadResult,
        *,
        provenance_path: Path | None = None,
    ) -> UploadResult:
        """Recheck an already uploaded payload before destructive cleanup.

        Custom upload adapters can override this method. The default preserves
        compatibility with lightweight test/fake adapters while requiring the
        caller to have an upload receipt marked verified.
        """
        if not upload.verified:
            raise RuntimeError("Cannot reverify an unverified upload")
        return upload


class KaggleCliUploader(DatasetUploader):
    """Use the official Kaggle CLI for create/version and status verification."""

    def __init__(
        self,
        executable: str,
        config: UploadConfig,
        *,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.executable = executable
        self.config = config
        self.progress = progress

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
        reused = self._reuse_matching_remote_payload(
            staging,
            dataset_ref=target.dataset_ref,
            mode=mode,
        )
        if reused is not None:
            return reused
        command = self._upload_command(
            staging.staging_dir,
            dataset_ref=target.dataset_ref,
            mode=mode,
        )
        return_code, output = self._run_streaming(
            command,
            desc=f"upload {target.dataset_ref}",
        )
        if return_code != 0:
            raise RuntimeError(f"Kaggle upload failed (exit {return_code}): {output[-3000:]}")
        if self.progress is not None:
            self.progress.update_activity(
                key="upload",
                desc=f"upload {target.dataset_ref}: verifying",
            )
        verification = self._verify_evidence(
            target.dataset_ref,
            expected_payload_digest=staging.payload_digest,
            provenance_path=staging.provenance_path,
        )
        if not verification["verified"]:
            if self.progress is not None:
                self.progress.update_activity(
                    key="upload",
                    desc=f"upload {target.dataset_ref}: verify failed",
                )
            raise RuntimeError(
                f"Kaggle upload was not verified: {verification['output'][-3000:]}"
            )
        if self.progress is not None:
            self.progress.complete_activity(
                key="upload",
                desc=f"upload {target.dataset_ref}: verified",
            )
        return UploadResult(
            dataset_ref=target.dataset_ref,
            mode=mode,
            verified=True,
            command=tuple(command),
            output_tail=output[-3000:],
            verified_output_tail=str(verification["output"])[-3000:],
            payload_digest=staging.payload_digest,
            remote_status=str(verification["status"]),
            remote_files=tuple(str(item) for item in verification["files"]),
            verified_at=str(verification["verified_at"]),
        )

    def _reuse_matching_remote_payload(
        self,
        staging: StagingResult,
        *,
        dataset_ref: str,
        mode: str,
    ) -> UploadResult | None:
        """Avoid uploading again after an interrupted post-transfer verification."""
        if (
            mode != "version"
            or staging.payload_digest is None
            or staging.provenance_path is None
            or not staging.provenance_path.is_file()
        ):
            return None
        if self.progress is not None:
            self.progress.start_activity(
                key="upload",
                desc=f"upload {dataset_ref}: checking remote",
                total=None,
                unit="B",
            )
        status_result = subprocess.run(
            self._status_command(dataset_ref),
            text=True,
            capture_output=True,
            check=False,
        )
        status_output = (status_result.stdout or "") + (status_result.stderr or "")
        if (
            status_result.returncode != 0
            or self._parse_ready_status(status_output) != "ready"
        ):
            return None
        remote_provenance, provenance_output = self._download_remote_provenance(
            dataset_ref
        )
        if (
            remote_provenance is None
            or remote_provenance.get("payload_digest") != staging.payload_digest
        ):
            return None
        if self.progress is not None:
            self.progress.complete_activity(
                key="upload",
                desc=f"upload {dataset_ref}: already verified",
            )
        return UploadResult(
            dataset_ref=dataset_ref,
            mode=mode,
            verified=True,
            command=(),
            output_tail="Matching Kaggle payload already exists; transfer skipped.",
            verified_output_tail=(status_output + provenance_output)[-3000:],
            payload_digest=staging.payload_digest,
            remote_status="ready",
            remote_files=(PROVENANCE_FILE_NAME,),
            verified_at=utc_now(),
        )

    def verify_existing(
        self,
        upload: UploadResult,
        *,
        provenance_path: Path | None = None,
    ) -> UploadResult:
        evidence = self._verify_evidence(
            upload.dataset_ref,
            expected_payload_digest=upload.payload_digest,
            provenance_path=provenance_path,
            audit_local_payload=True,
        )
        if not evidence["verified"]:
            raise RuntimeError(
                f"Remote Kaggle verification failed before cleanup: {evidence['output'][-3000:]}"
            )
        return UploadResult(
            dataset_ref=upload.dataset_ref,
            mode=upload.mode,
            verified=True,
            command=upload.command,
            output_tail=upload.output_tail,
            verified_output_tail=str(evidence["output"])[-3000:],
            payload_digest=upload.payload_digest,
            remote_status=str(evidence["status"]),
            remote_files=tuple(str(item) for item in evidence["files"]),
            verified_at=str(evidence["verified_at"]),
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
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode == 0:
            return "version"
        lowered = output.lower()
        if any(
            marker in lowered
            for marker in (
                "not found",
                "does not exist",
                "not exist",
                "403 client error",
                "forbidden",
            )
        ):
            return "create"
        raise RuntimeError(
            "Could not determine whether the Kaggle dataset exists; "
            f"status command failed with exit {completed.returncode}: {output[-2000:]}"
        )

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
            # Kaggle CLI defaults newly-created datasets to private.  Its
            # supported visibility flag is --public; --private is rejected by
            # the CLI versions commonly installed on SSH hosts.
            if self.config.public:
                command.append("--public")
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
        """Compatibility wrapper returning the textual verification evidence."""
        effective_ref = dataset_ref or self.config.dataset_ref
        if not effective_ref:
            return False, "dataset_ref is required for remote verification"
        command = self._status_command(effective_ref)
        deadline = time.monotonic() + self.config.verify_timeout_seconds
        output = ""
        while time.monotonic() <= deadline:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            output = (completed.stdout or "") + (completed.stderr or "")
            if completed.returncode == 0 and self._parse_ready_status(output) == "ready":
                return True, output
            time.sleep(self.config.verify_poll_seconds)
        return False, output

    def _verify_evidence(
        self,
        dataset_ref: str | None = None,
        *,
        expected_payload_digest: str | None = None,
        provenance_path: Path | None = None,
        audit_local_payload: bool = False,
    ) -> dict[str, Any]:
        effective_ref = dataset_ref or self.config.dataset_ref
        if not effective_ref:
            return {
                "verified": False,
                "status": "missing_ref",
                "files": (),
                "output": "dataset_ref is required for remote verification",
                "verified_at": None,
            }
        status_command = self._status_command(effective_ref)
        deadline = time.monotonic() + self.config.verify_timeout_seconds
        output = ""
        last_status = "timeout"
        local_provenance: dict[str, Any] | None = None
        if provenance_path is not None:
            try:
                payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return {
                    "verified": False,
                    "status": "invalid_local_provenance",
                    "files": (),
                    "output": f"Could not read local provenance: {exc}",
                    "verified_at": None,
                }
            if not isinstance(payload, Mapping):
                return {
                    "verified": False,
                    "status": "invalid_local_provenance",
                    "files": (),
                    "output": "Local provenance must contain one JSON object",
                    "verified_at": None,
                }
            local_provenance = dict(payload)
            if (
                expected_payload_digest is not None
                and local_provenance.get("payload_digest") != expected_payload_digest
            ):
                return {
                    "verified": False,
                    "status": "local_payload_digest_mismatch",
                    "files": (),
                    "output": "Local payload digest does not match provenance",
                    "verified_at": None,
                }
            if audit_local_payload:
                records = local_provenance.get("files")
                if not isinstance(records, list) or not all(
                    isinstance(record, Mapping) for record in records
                ):
                    return {
                        "verified": False,
                        "status": "invalid_local_inventory",
                        "files": (),
                        "output": "Local provenance has no valid file inventory",
                        "verified_at": None,
                    }
                if not self._local_inventory_matches(provenance_path.parent, records):
                    return {
                        "verified": False,
                        "status": "local_payload_changed",
                        "files": (),
                        "output": "Local staging payload files changed after upload",
                        "verified_at": None,
                    }
        while time.monotonic() <= deadline:
            status_result = subprocess.run(
                status_command,
                text=True,
                capture_output=True,
                check=False,
            )
            status_output = (status_result.stdout or "") + (status_result.stderr or "")
            status = self._parse_ready_status(status_output) if status_result.returncode == 0 else None
            if status_result.returncode == 0 and status == "ready":
                if provenance_path is None or not self.config.require_remote_inventory:
                    return {
                        "verified": True,
                        "status": status,
                        "files": (),
                        "output": status_output,
                        "verified_at": utc_now(),
                    }
                remote_provenance, provenance_output = self._download_remote_provenance(
                    effective_ref
                )
                output = status_output + provenance_output
                if remote_provenance is None:
                    last_status = "ready_missing_provenance"
                elif remote_provenance.get("payload_digest") != (
                    expected_payload_digest
                    or (
                        local_provenance.get("payload_digest")
                        if local_provenance is not None
                        else None
                    )
                ):
                    last_status = "ready_provenance_digest_mismatch"
                    output += "\nremote payload digest does not match uploaded payload"
                else:
                    return {
                        "verified": True,
                        "status": status,
                        "files": (PROVENANCE_FILE_NAME,),
                        "output": output,
                        "verified_at": utc_now(),
                    }
            else:
                output = status_output
                last_status = "status_not_ready"
            time.sleep(self.config.verify_poll_seconds)
        return {
            "verified": False,
            "status": last_status,
            "files": (),
            "output": output,
            "verified_at": None,
        }

    def _download_remote_provenance(
        self,
        dataset_ref: str,
    ) -> tuple[dict[str, Any] | None, str]:
        """Download exact remote evidence instead of scanning only one file page."""
        with tempfile.TemporaryDirectory(prefix="preprocess-kaggle-verify-") as temporary:
            destination = Path(temporary)
            command = [
                self.executable,
                "datasets",
                "download",
                dataset_ref,
                "--file",
                PROVENANCE_FILE_NAME,
                "--path",
                str(destination),
                "--force",
                "--quiet",
            ]
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            if completed.returncode != 0:
                return None, output
            candidates = tuple(destination.rglob(PROVENANCE_FILE_NAME))
            if len(candidates) != 1:
                return None, output + "\nremote provenance.json was not downloaded uniquely"
            try:
                payload = json.loads(candidates[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return None, output + f"\ninvalid remote provenance.json: {exc}"
            if not isinstance(payload, Mapping):
                return None, output + "\nremote provenance.json is not a JSON object"
            return dict(payload), output

    @staticmethod
    def _local_inventory_matches(
        staging_root: Path,
        records: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Full-audit allowlisted staging files without following paths outside it."""
        try:
            root = staging_root.resolve()
            for record in records:
                raw_path = record.get("path")
                if not isinstance(raw_path, str) or not raw_path:
                    return False
                candidate = (root / raw_path).resolve()
                candidate.relative_to(root)
                if not file_fingerprint_matches(candidate, record, full_audit=True):
                    return False
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    def _run_streaming(
        self,
        command: Sequence[str],
        *,
        desc: str = "Kaggle upload",
    ) -> tuple[int, str]:
        """Run Kaggle with bounded output and publish its transfer progress."""
        if self.progress is not None:
            self.progress.start_activity(
                key="upload",
                desc=f"{desc}: preparing",
                total=None,
                unit="B",
            )
        try:
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Required executable was not found: {command[0]}") from exc
        assert process.stdout is not None
        tail = bytearray()
        progress_buffer = ""
        transfer_total: int | None = None
        transfer_current = 0
        transfer_part = 0

        def publish_progress(line: str) -> None:
            nonlocal transfer_total, transfer_current, transfer_part
            matches = tuple(_KAGGLE_TRANSFER_PROGRESS.finditer(line))
            if not matches or self.progress is None:
                return
            match = matches[-1]
            current = _scaled_transfer_bytes(
                match.group("current"), match.group("current_unit")
            )
            total = _scaled_transfer_bytes(
                match.group("total"), match.group("total_unit")
            )
            if total <= 0:
                return
            if transfer_total != total or current < transfer_current:
                transfer_part += 1
                self.progress.start_activity(
                    key="upload",
                    desc=f"{desc} part={transfer_part}",
                    total=total,
                    unit="B",
                )
            transfer_total = total
            transfer_current = min(current, total)
            self.progress.update_activity(
                key="upload",
                current=transfer_current,
                total=transfer_total,
                desc=f"{desc} part={transfer_part}",
            )

        read_chunk = getattr(process.stdout, "read1", process.stdout.read)
        while chunk := read_chunk(16 * 1024):
            tail.extend(chunk)
            if len(tail) > 12_000:
                del tail[:-12_000]
            progress_buffer += chunk.decode("utf-8", errors="replace")
            rows = re.split(r"[\r\n]", progress_buffer)
            progress_buffer = rows.pop()
            for row in rows:
                publish_progress(row)
        if progress_buffer:
            publish_progress(progress_buffer)
        return_code = process.wait()
        if self.progress is not None:
            if return_code == 0:
                self.progress.complete_activity(key="upload", desc=f"{desc}: sent")
            else:
                self.progress.update_activity(key="upload", desc=f"{desc}: failed")
        return return_code, bytes(tail).decode("utf-8", errors="replace")

    @staticmethod
    def _parse_ready_status(output: str) -> str | None:
        """Parse the legacy/current human-readable Kaggle status output."""
        for raw_line in output.splitlines():
            line = raw_line.strip().lower()
            if line == "ready" or line in {"status: ready", "status=ready"}:
                return "ready"
        return None
