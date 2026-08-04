"""Filesystem layout for one archive-derived lot.

The layout is intentionally independent from the current working directory;
callers provide ``data_root``.  This keeps SSH deployments portable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LotLayout:
    data_root: Path
    lot_id: str

    def __post_init__(self) -> None:
        if not self.lot_id or self.lot_id in {".", ".."} or Path(self.lot_id).name != self.lot_id:
            raise ValueError(f"Unsafe lot_id for filesystem layout: {self.lot_id!r}")

    @property
    def root(self) -> Path:
        return self.data_root / self.lot_id

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def source_root(self) -> Path:
        return self.source_dir / self.lot_id

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def staging_dir(self) -> Path:
        return self.root / "kaggle-staging"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def receipts_dir(self) -> Path:
        return self.root / "receipts"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def upload_state_path(self) -> Path:
        return self.root / "upload-state.json"

    @property
    def upload_lock_path(self) -> Path:
        return self.root / "upload.lock"

    def create_runtime_dirs(self) -> None:
        """Create only the directories owned by this lot."""
        for directory in (
            self.archive_dir,
            self.source_dir,
            self.dataset_dir,
            self.reports_dir,
            self.receipts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def is_owned_path(self, path: Path) -> bool:
        """Return whether ``path`` is inside this lot, never the lot itself."""
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        return path.resolve() != self.root.resolve()
