"""Stable data contracts shared by the batch pipeline stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    """Return a JSON-friendly UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class BatchState(str, Enum):
    """Durable states for one archive/lot workflow."""

    NEW = "new"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    ARCHIVE_VALIDATED = "archive_validated"
    EXTRACTED = "extracted"
    VIDEOS_DISCOVERED = "videos_discovered"
    SHOT_BOUNDARIES = "shot_boundaries"
    SHOT_BOUNDARIES_READY = "shot_boundaries_ready"
    PROCESSING = "processing"
    PROCESSED = "processed"
    VALIDATED = "validated"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    STAGED = "staged"
    UPLOADING = "uploading"
    UPLOADED_VERIFIED = "uploaded_verified"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ArchiveInput:
    """One direct archive URL from the user-provided links file."""

    url: str
    archive_name: str
    lot_id: str
    line_number: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one downloader invocation."""

    request: ArchiveInput
    path: Path
    command: tuple[str, ...]
    resumed: bool
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        value["command"] = list(self.command)
        return value


@dataclass(frozen=True)
class ArchiveInspection:
    """Validated ZIP inventory before extraction."""

    archive_path: Path
    archive_size_bytes: int
    root_name: str
    members: tuple[str, ...]
    video_members: tuple[str, ...]
    uncompressed_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["archive_path"] = str(self.archive_path)
        value["members"] = list(self.members)
        value["video_members"] = list(self.video_members)
        return value


@dataclass(frozen=True)
class VideoAsset:
    """A video discovered inside one renamed archive root."""

    video_id: str
    path: Path
    lot_id: str
    source_name: str

    def __post_init__(self) -> None:
        if not self.video_id or self.video_id in {".", ".."} or Path(self.video_id).name != self.video_id:
            raise ValueError(f"Unsafe video_id: {self.video_id!r}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        return value


@dataclass(frozen=True)
class ProcessingResult:
    """Artifacts produced for one video by an injected processing strategy."""

    asset: VideoAsset
    video_info: Mapping[str, Any]
    selected_count: int
    selection_manifest_path: Path
    rendered_manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["asset"] = self.asset.to_dict()
        value["selection_manifest_path"] = str(self.selection_manifest_path)
        value["rendered_manifest_path"] = str(self.rendered_manifest_path)
        return value


@dataclass(frozen=True)
class StagingResult:
    """Manifest of the exact local payload prepared for an uploader."""

    staging_dir: Path
    files: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"staging_dir": str(self.staging_dir), "files": [str(path) for path in self.files]}


@dataclass(frozen=True)
class UploadResult:
    """Remote upload acknowledgement and verification evidence."""

    dataset_ref: str
    mode: str
    verified: bool
    command: tuple[str, ...]
    output_tail: str
    verified_output_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        return value


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable validation finding."""

    code: str
    message: str
    severity: str = "error"
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Collected validation evidence for a lot or a single video."""

    subject: str
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add_error(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(ValidationIssue(code, message, "error", context))

    def add_warning(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(ValidationIssue(code, message, "warning", context))

    def finish(self) -> "ValidationReport":
        self.finished_at = utc_now()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "passed": self.passed,
            "measurements": self.measurements,
            "issues": [issue.to_dict() for issue in self.issues],
        }
