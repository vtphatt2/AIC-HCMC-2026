"""Stable data contracts shared by keyframe selectors and extractors.

The contracts deliberately keep selection independent from image encoding and
filesystem layout.  That lets a selection manifest be rendered repeatedly with
different image profiles without re-running the selection algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class VideoSource:
    """The caller-owned input location and identity for one video."""

    video_id: str
    path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoInfo:
    """Properties measured from the media file, not inferred from metadata."""

    video_id: str
    duration_ms: int
    fps: float | None
    width: int
    height: int
    frame_count: int | None
    codec: str | None


@dataclass(frozen=True)
class FrameRef:
    """A decoded presentation-order frame.

    ``source_frame_number`` is zero-based and is only an identifier.  PTS-derived
    ``timestamp_ms`` remains the timing source of truth, including for VFR video.
    """

    video_id: str
    source_frame_number: int
    timestamp_ms: int
    pts_time_seconds: float

    @property
    def frame_id(self) -> str:
        return f"{self.source_frame_number:06d}"


@dataclass(frozen=True)
class FrameCandidate:
    """A frame that a selector may evaluate; preview/features are optional."""

    ref: FrameRef
    preview: Any | None = None
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectedFrame:
    """A selector decision, with audit information retained in the manifest."""

    ref: FrameRef
    score: float
    reasons: tuple[str, ...]
    rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderProfile:
    """Image rendering configuration.

    A ``None`` short edge preserves the decoded frame dimensions. When a
    positive short edge is supplied, the image is resized proportionally.
    """

    profile_id: str
    target_short_edge_px: int | None = None
    image_format: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = 100
    png_compress_level: int = 6
    allow_upscale: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty")
        if self.target_short_edge_px is not None and self.target_short_edge_px <= 0:
            raise ValueError("target_short_edge_px must be positive when provided")
        if self.image_format not in {"png", "jpeg"}:
            raise ValueError("image_format must be 'png' or 'jpeg'")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0 <= self.png_compress_level <= 9:
            raise ValueError("png_compress_level must be between 0 and 9")

    @property
    def file_extension(self) -> str:
        return ".png" if self.image_format == "png" else ".jpg"


@dataclass(frozen=True)
class SceneSegment:
    """A half-open scene interval supplied by an external scene detector."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("SceneSegment requires 0 <= start_ms < end_ms")


@dataclass(frozen=True)
class ExtractionOutput:
    """Locations and output behaviour supplied by the orchestration layer."""

    rendered_root: Path
    selection_manifest_path: Path
    profile: RenderProfile
    overwrite: bool = False


@dataclass(frozen=True)
class ExtractedFrame:
    ref: FrameRef
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class ExtractionResult:
    video_id: str
    written: Sequence[ExtractedFrame]
    skipped: Sequence[FrameRef]
    rendered_manifest_path: Path


class KeyframeSelector(Protocol):
    """Pure decision boundary: implementations must not write output files."""

    name: str
    version: str

    def select(
        self,
        video: VideoInfo,
        candidates: Iterable[FrameCandidate],
    ) -> Sequence[SelectedFrame]:
        ...


class KeyframeExtractor(Protocol):
    """Video I/O boundary: implementations decode media and write image output."""

    name: str
    version: str

    def probe(self, source: VideoSource) -> VideoInfo:
        ...

    def scan(self, source: VideoSource, video: VideoInfo) -> Iterable[FrameCandidate]:
        ...

    def materialize(
        self,
        source: VideoSource,
        video: VideoInfo,
        selected: Sequence[SelectedFrame],
        output: ExtractionOutput,
    ) -> ExtractionResult:
        ...
