"""OOP building blocks for the SSH-friendly preprocessing pipeline.

The package deliberately contains no repository-specific absolute paths.  A
``BatchConfig`` supplies all locations and external command names at runtime.
"""

from preprocess.batch.layout import LotLayout
from preprocess.batch.models import (
    ArchiveInput,
    ArchiveInspection,
    BatchState,
    DownloadResult,
    ProcessingResult,
    StagingResult,
    UploadResult,
    ValidationIssue,
    ValidationReport,
    VideoAsset,
)

__all__ = [
    "ArchiveInput",
    "ArchiveInspection",
    "BatchState",
    "DownloadResult",
    "LotLayout",
    "ProcessingResult",
    "StagingResult",
    "UploadResult",
    "ValidationIssue",
    "ValidationReport",
    "VideoAsset",
]
