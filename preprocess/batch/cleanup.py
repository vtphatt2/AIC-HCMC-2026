"""Manifest-driven cleanup that can only run after verified upload."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preprocess.batch.config import CleanupConfig
from preprocess.batch.layout import LotLayout
from preprocess.batch.models import UploadResult, utc_now
from preprocess.batch.provenance import atomic_json_write


@dataclass(frozen=True)
class CleanupResult:
    deleted: tuple[str, ...]
    skipped: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Any) -> "CleanupResult":
        if not isinstance(payload, dict):
            raise ValueError("Cleanup receipt must be a JSON object")
        deleted = payload.get("deleted", [])
        skipped = payload.get("skipped", [])
        if not isinstance(deleted, list) or not isinstance(skipped, list):
            raise ValueError("Cleanup receipt has invalid deleted/skipped lists")
        return cls(
            deleted=tuple(str(item) for item in deleted),
            skipped=tuple(str(item) for item in skipped),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"deleted": list(self.deleted), "skipped": list(self.skipped)}


class CleanupManager:
    """Delete only explicitly configured, lot-owned artifact directories."""

    def __init__(
        self,
        config: CleanupConfig,
        *,
        rendered_profile_id: str = "keyframes",
        preserve_staging: bool = False,
    ) -> None:
        self.config = config
        self.preserve_staging = preserve_staging
        if (
            not rendered_profile_id
            or rendered_profile_id in {".", ".."}
            or Path(rendered_profile_id).name != rendered_profile_id
        ):
            raise ValueError(f"Unsafe rendered profile id: {rendered_profile_id!r}")
        self.rendered_profile_id = rendered_profile_id

    def cleanup(self, layout: LotLayout, upload: UploadResult) -> CleanupResult:
        if not upload.verified:
            raise RuntimeError("Cleanup requires a verified upload")
        if not self.config.enabled:
            result = CleanupResult(deleted=(), skipped=("cleanup disabled",))
            self._write_receipt(layout, result)
            return result

        targets: list[tuple[str, Path, bool]] = [
            ("archive", layout.archive_dir, self.config.delete_archive),
            ("source", layout.source_dir, self.config.delete_source),
            (
                "keyframes",
                layout.dataset_dir / self.rendered_profile_id,
                self.config.delete_keyframes,
            ),
            ("features", layout.dataset_dir / "PECore-features", self.config.delete_features),
            (
                "manifests",
                layout.dataset_dir / "selection-manifests",
                self.config.delete_manifests,
            ),
            (
                "transcripts",
                layout.dataset_dir / "transcripts",
                self.config.delete_transcripts,
            ),
            (
                "transcript-index",
                layout.dataset_dir / "keyframe_transcript_index",
                self.config.delete_transcript_index,
            ),
        ]
        deleted: list[str] = []
        skipped: list[str] = []
        for label, path, enabled in targets:
            if not enabled:
                skipped.append(f"{label}: disabled")
                continue
            if not path.exists():
                skipped.append(f"{label}: absent")
                continue
            if not layout.is_owned_path(path):
                raise RuntimeError(f"Refusing to delete path outside lot: {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(str(path))
        if self.preserve_staging:
            skipped.append("kaggle-staging: preserved for cumulative dataset")
        else:
            staging_path = layout.staging_dir
            if not self.config.delete_staging:
                skipped.append("kaggle-staging: disabled")
            elif not staging_path.exists():
                skipped.append("kaggle-staging: absent")
            else:
                if not layout.is_owned_path(staging_path):
                    raise RuntimeError(f"Refusing to delete path outside lot: {staging_path}")
                if staging_path.is_dir():
                    shutil.rmtree(staging_path)
                else:
                    staging_path.unlink()
                deleted.append(str(staging_path))
        result = CleanupResult(deleted=tuple(deleted), skipped=tuple(skipped))
        self._write_receipt(layout, result)
        return result

    @staticmethod
    def _write_receipt(layout: LotLayout, result: CleanupResult) -> None:
        path = layout.receipts_dir / "cleanup.json"
        payload = {"finished_at": utc_now(), **result.to_dict()}
        atomic_json_write(path, payload)
