"""Durable state for cumulative dataset staging and uploads."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from preprocess.batch.models import UploadResult, utc_now
from preprocess.batch.provenance import PROVENANCE_FILE_NAME, atomic_json_write, digest_directory


class DatasetUploadStateStore:
    """Persist which lots are present in the cumulative dataset snapshot.

    This state is deliberately separate from each lot's ``state.json`` and
    ``upload-state.json``.  A lot checkpoint describes one lot; this document
    describes the dataset-wide snapshot that is sent to Kaggle.
    """

    schema_version = 1

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": self.schema_version,
                "initialized": False,
                "lots": {},
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Dataset upload state must be a JSON object: {self.path}")
        payload.setdefault("schema_version", self.schema_version)
        payload.setdefault("lots", {})
        return payload

    def initialize(self, *, dataset_ref: str, staging_dir: Path) -> dict[str, Any]:
        state = self.load()
        if state.get("initialized"):
            previous_ref = state.get("dataset_ref")
            previous_staging = state.get("staging_dir")
            if previous_ref and previous_ref != dataset_ref:
                raise RuntimeError(
                    "Cumulative dataset state belongs to a different dataset: "
                    f"{previous_ref!r} != {dataset_ref!r}"
                )
            if previous_staging and Path(str(previous_staging)).resolve() != staging_dir.resolve():
                raise RuntimeError(
                    "Cumulative dataset state points to a different staging directory: "
                    f"{previous_staging!r} != {staging_dir}"
                )
        else:
            state.update(
                {
                    "schema_version": self.schema_version,
                    "initialized": True,
                    "dataset_ref": dataset_ref,
                    "staging_dir": str(staging_dir),
                    "created_at": utc_now(),
                    "lots": {},
                }
            )
        state["updated_at"] = utc_now()
        self.write(state)
        return state

    def ensure_staging_available(self, staging_dir: Path) -> None:
        state = self.load()
        lots = state.get("lots", {})
        has_files = staging_dir.is_dir() and any(path.is_file() for path in staging_dir.rglob("*"))
        if lots and not has_files:
            raise RuntimeError(
                "Cumulative staging is missing while dataset state contains uploaded lots: "
                f"{staging_dir}. Restore it or start a new dataset state before uploading."
            )
        if not lots:
            return
        provenance_path = staging_dir / PROVENANCE_FILE_NAME
        if not provenance_path.is_file():
            raise RuntimeError(
                "Cumulative staging has no provenance.json while dataset state contains lots: "
                f"{staging_dir}"
            )
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            expected_digest = str(provenance["payload_digest"])
            actual_digest, _ = digest_directory(
                staging_dir,
                exclude_names={PROVENANCE_FILE_NAME},
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cumulative staging provenance is invalid: {provenance_path}") from exc
        if expected_digest != actual_digest:
            raise RuntimeError(
                "Cumulative staging payload digest does not match provenance; "
                "refusing to merge into a corrupted snapshot"
            )

    def record_staged(
        self,
        lot_id: str,
        *,
        video_ids: Sequence[str],
        files: Sequence[str],
        payload_digest: str | None = None,
    ) -> dict[str, Any]:
        state = self.load()
        lots = state.setdefault("lots", {})
        if not isinstance(lots, dict):
            raise ValueError(f"Invalid lots map in dataset upload state: {self.path}")
        previous = lots.get(lot_id, {})
        lots[lot_id] = {
            **(dict(previous) if isinstance(previous, Mapping) else {}),
            "status": "staged",
            "video_ids": sorted({str(video_id) for video_id in video_ids}),
            "files": sorted({str(file) for file in files}),
            "payload_digest": payload_digest,
            "staged_at": utc_now(),
        }
        state["updated_at"] = utc_now()
        self.write(state)
        return state

    def record_uploaded(self, lot_id: str, upload: UploadResult) -> dict[str, Any]:
        state = self.load()
        lots = state.setdefault("lots", {})
        if not isinstance(lots, dict):
            raise ValueError(f"Invalid lots map in dataset upload state: {self.path}")
        previous = lots.get(lot_id, {})
        lots[lot_id] = {
            **(dict(previous) if isinstance(previous, Mapping) else {}),
            "status": "uploaded",
            "uploaded_at": utc_now(),
            "upload": upload.to_dict(),
        }
        state["last_upload"] = {
            "lot_id": lot_id,
            "at": utc_now(),
            "upload": upload.to_dict(),
        }
        state["updated_at"] = utc_now()
        self.write(state)
        return state

    def write(self, payload: Mapping[str, Any]) -> None:
        atomic_json_write(self.path, payload)


def dataset_state_path(staging_dir: Path) -> Path:
    """Return the local state path next to a cumulative staging directory."""
    return staging_dir.parent / "kaggle-dataset-state.json"


def dataset_upload_lock_path(staging_dir: Path) -> Path:
    """Return the dataset-wide lock path next to a cumulative staging directory."""
    return staging_dir.parent / "kaggle-dataset-upload.lock"
