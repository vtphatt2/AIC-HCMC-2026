"""Manifest-driven cleanup that can only run after verified upload."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preprocess.batch.config import CleanupConfig
from preprocess.batch.layout import LotLayout
from preprocess.batch.models import UploadResult, utc_now


@dataclass(frozen=True)
class CleanupResult:
    deleted: tuple[str, ...]
    skipped: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"deleted": list(self.deleted), "skipped": list(self.skipped)}


class CleanupManager:
    """Delete only explicitly configured, lot-owned artifact directories."""

    def __init__(self, config: CleanupConfig, *, rendered_profile_id: str = "keyframes") -> None:
        self.config = config
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
            ("kaggle-staging", layout.staging_dir, self.config.delete_staging),
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
        result = CleanupResult(deleted=tuple(deleted), skipped=tuple(skipped))
        self._write_receipt(layout, result)
        return result

    @staticmethod
    def _write_receipt(layout: LotLayout, result: CleanupResult) -> None:
        path = layout.receipts_dir / "cleanup.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"finished_at": utc_now(), **result.to_dict()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
