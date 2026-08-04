"""Archive extraction adapters with atomic root-directory rename."""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath

from preprocess.batch.models import ArchiveInspection
from preprocess.batch.provenance import sha256_file


class ArchiveExtractor(ABC):
    """Replaceable boundary for archive formats."""

    @abstractmethod
    def extract(
        self,
        inspection: ArchiveInspection,
        destination_root: Path,
        lot_id: str,
    ) -> Path:
        raise NotImplementedError


class ZipArchiveExtractor(ArchiveExtractor):
    """Extract a validated ZIP and rename its ``video`` root to the lot ID."""

    def extract(self, inspection: ArchiveInspection, destination_root: Path, lot_id: str) -> Path:
        if not inspection.archive_path.is_file():
            raise FileNotFoundError(f"Archive not found during extraction: {inspection.archive_path}")
        if inspection.archive_sha256 is not None:
            current_digest = sha256_file(inspection.archive_path)
            if current_digest != inspection.archive_sha256:
                raise RuntimeError(
                    "Archive changed after validation; refusing to extract a different payload"
                )
        destination_root.mkdir(parents=True, exist_ok=True)
        target = destination_root / lot_id
        if target.exists():
            raise FileExistsError(f"Extraction target already exists: {target}")

        temporary_root = Path(tempfile.mkdtemp(prefix=f".{lot_id}.", dir=destination_root))
        try:
            with zipfile.ZipFile(inspection.archive_path) as archive:
                for info in archive.infolist():
                    relative = PurePosixPath(info.filename.replace("\\", "/"))
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Unsafe ZIP member path during extraction: {info.filename}")
                    destination = temporary_root.joinpath(*relative.parts)
                    if info.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info, "r") as source, destination.open("wb") as target_file:
                        shutil.copyfileobj(source, target_file)

            extracted_root = temporary_root / inspection.root_name
            if not extracted_root.is_dir():
                raise ValueError(f"Expected extracted root is missing: {extracted_root}")
            os.replace(extracted_root, target)
            return target
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
