"""Configuration objects for portable SSH batch runs.

Configuration is JSON-based so the pipeline does not need another Python
dependency.  Paths in a config file are resolved relative to that file.
Secrets are intentionally not accepted here; Kaggle credentials remain in the
normal Kaggle CLI environment/configuration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from preprocess.keyframes.contracts import RenderProfile
from preprocess.pecore.embedding import PECoreEmbeddingConfig
from preprocess.progress import ProgressConfig


def _resolve_path(value: str | Path | None, base_dir: Path, default: Path | None = None) -> Path | None:
    if value is None:
        if default is None:
            return None
        path = Path(default).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


@dataclass(frozen=True)
class ToolConfig:
    aria2c: str = "aria2c"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    kaggle: str = "kaggle"


SCENE_BOUNDARY_SELECTORS = frozenset({"scene-segments", "linear-rulebase", "linear"})


def selector_requires_scene_boundaries(selector: str) -> bool:
    """Return whether a selector consumes detector-produced shot boundaries."""
    return selector.strip().lower() in SCENE_BOUNDARY_SELECTORS


@dataclass(frozen=True)
class DownloadConfig:
    continue_download: bool = True
    auto_file_renaming: bool = False
    max_tries: int = 5
    retry_wait_seconds: int = 5
    timeout_seconds: int = 60
    connect_timeout_seconds: int = 30
    max_concurrent_connections: int = 16
    split_count: int = 16
    check_integrity: bool = False

    def __post_init__(self) -> None:
        if self.max_tries <= 0:
            raise ValueError("max_tries must be positive")
        if self.retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must not be negative")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("download timeouts must be positive")
        if self.max_concurrent_connections <= 0:
            raise ValueError("max_concurrent_connections must be positive")
        if self.split_count <= 0:
            raise ValueError("split_count must be positive")


@dataclass(frozen=True)
class ArchiveConfig:
    prefix: str = "Videos_"
    extension: str = ".zip"
    expected_root_name: str = "video"
    video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".mov", ".avi", ".webm")
    max_members: int | None = None
    max_uncompressed_bytes: int | None = None
    minimum_archive_bytes: int = 1

    def __post_init__(self) -> None:
        normalized = tuple(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in self.video_extensions
        )
        object.__setattr__(self, "video_extensions", normalized)
        if not self.extension.startswith("."):
            object.__setattr__(self, "extension", f".{self.extension}")
        if not self.expected_root_name or "/" in self.expected_root_name or "\\" in self.expected_root_name:
            raise ValueError("expected_root_name must be one directory name")
        if self.max_members is not None and self.max_members <= 0:
            raise ValueError("max_members must be positive when provided")
        if self.max_uncompressed_bytes is not None and self.max_uncompressed_bytes <= 0:
            raise ValueError("max_uncompressed_bytes must be positive when provided")
        if self.minimum_archive_bytes < 0:
            raise ValueError("minimum_archive_bytes must not be negative")


@dataclass(frozen=True)
class LinearSelectionConfig:
    """Configurable piecewise-linear keyframe count rule per scene/shot."""

    short_duration_ms: int = 1_000
    short_frame_count: int = 1
    base_duration_ms: int = 3_000
    base_frame_count: int = 2
    increment_duration_ms: int = 3_000
    increment_frame_count: int = 1

    def __post_init__(self) -> None:
        if self.short_duration_ms <= 0 or self.base_duration_ms <= 0:
            raise ValueError("linear duration thresholds must be positive")
        if self.base_duration_ms <= self.short_duration_ms:
            raise ValueError("base_duration_ms must be greater than short_duration_ms")
        if self.increment_duration_ms <= 0:
            raise ValueError("increment_duration_ms must be positive")
        if self.short_frame_count <= 0 or self.base_frame_count <= 0:
            raise ValueError("linear frame counts must be positive")
        if self.base_frame_count < self.short_frame_count:
            raise ValueError("base_frame_count must not be less than short_frame_count")
        if self.increment_frame_count <= 0:
            raise ValueError("increment_frame_count must be positive")

    def frame_count(self, duration_ms: int) -> int:
        """Return the configured number of samples for one scene duration."""
        if duration_ms <= 0:
            raise ValueError("scene duration must be positive")
        if duration_ms <= self.short_duration_ms:
            return self.short_frame_count
        if duration_ms <= self.base_duration_ms:
            return self.base_frame_count
        increments = (
            duration_ms - self.base_duration_ms + self.increment_duration_ms - 1
        ) // self.increment_duration_ms
        return self.base_frame_count + increments * self.increment_frame_count


@dataclass(frozen=True)
class ShotBoundaryConfig:
    """Automatic detector settings for selectors that operate per shot/scene."""

    enabled: bool = True
    backend: str = "transnetv2"
    output_dir: Path | None = None
    device: str = "auto"
    threshold: float = 0.5
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("shot boundary backend must not be empty")
        if not self.device.strip():
            raise ValueError("shot boundary device must not be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("shot boundary threshold must be between 0 and 1")


@dataclass(frozen=True)
class ProcessingConfig:
    strategy: str = "keyframes"
    selector: str = "uniform"
    interval_ms: int = 1_000
    scene_segments_dir: Path | None = None
    linear_rule: LinearSelectionConfig = field(default_factory=LinearSelectionConfig)
    profile_id: str = "keyframes"
    target_short_edge_px: int | None = None
    image_format: str = "png"
    jpeg_quality: int = 100
    png_compress_level: int = 6
    allow_upscale: bool = False
    overwrite: bool = False
    decode_checkpoints: tuple[float, ...] = (0.0, 0.5, 1.0)

    def render_profile(self) -> RenderProfile:
        return RenderProfile(
            profile_id=self.profile_id,
            target_short_edge_px=self.target_short_edge_px,
            image_format=self.image_format,  # type: ignore[arg-type]
            jpeg_quality=self.jpeg_quality,
            png_compress_level=self.png_compress_level,
            allow_upscale=self.allow_upscale,
        )

    def __post_init__(self) -> None:
        if not self.strategy.strip():
            raise ValueError("processing strategy must not be empty")
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if self.target_short_edge_px is not None and self.target_short_edge_px <= 0:
            raise ValueError("target_short_edge_px must be positive when provided")
        if (
            not self.profile_id.strip()
            or self.profile_id in {".", ".."}
            or Path(self.profile_id).name != self.profile_id
        ):
            raise ValueError("profile_id must be one safe directory name")
        if self.image_format not in {"jpeg", "png"}:
            raise ValueError("image_format must be 'jpeg' or 'png'")
        if not self.decode_checkpoints:
            raise ValueError("decode_checkpoints must not be empty")
        if any(not 0 <= point <= 1 for point in self.decode_checkpoints):
            raise ValueError("decode_checkpoints must be between 0 and 1")


@dataclass(frozen=True)
class UploadConfig:
    enabled: bool = False
    dataset_ref: str | None = None
    mode: str = "version"
    metadata_template: Path | None = None
    include_features: bool = True
    include_transcripts: bool = True
    include_transcript_index: bool = True
    version_message: str = "preprocess batch upload"
    public: bool = False
    verify_timeout_seconds: int = 600
    verify_poll_seconds: int = 10

    def __post_init__(self) -> None:
        if self.mode not in {"create", "version"}:
            raise ValueError("upload mode must be 'create' or 'version'")
        if self.enabled and not self.dataset_ref:
            raise ValueError("dataset_ref is required for a verifiable upload")
        if self.verify_timeout_seconds <= 0 or self.verify_poll_seconds <= 0:
            raise ValueError("upload verification timings must be positive")


@dataclass(frozen=True)
class CleanupConfig:
    enabled: bool = True
    delete_archive: bool = True
    delete_source: bool = True
    delete_keyframes: bool = True
    delete_features: bool = True
    delete_manifests: bool = True
    delete_transcripts: bool = True
    delete_transcript_index: bool = True
    delete_staging: bool = True


@dataclass(frozen=True)
class BatchConfig:
    """Top-level settings with no user-specific absolute paths."""

    data_root: Path = Path("data")
    links_file: Path = Path("data/input/links.txt")
    metadata_root: Path = Path("data/metadata")
    tools: ToolConfig = field(default_factory=ToolConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    shot_boundary: ShotBoundaryConfig = field(default_factory=ShotBoundaryConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    embedding: PECoreEmbeddingConfig = field(default_factory=PECoreEmbeddingConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    minimum_free_bytes: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "BatchConfig":
        """Load config and resolve relative paths against the config location."""
        path = path.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("batch config must be a JSON object")
        base_dir = path.parent
        return cls.from_mapping(payload, base_dir=base_dir)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: Path = Path(".")) -> "BatchConfig":
        tools = ToolConfig(**dict(payload.get("tools", {})))
        download = DownloadConfig(**dict(payload.get("download", {})))
        progress = ProgressConfig(**dict(payload.get("progress", {})))
        archive = ArchiveConfig(**dict(payload.get("archive", {})))

        data_root = _resolve_path(payload.get("data_root"), base_dir, Path("data")) or Path("data")

        processing_payload = dict(payload.get("processing", {}))
        processing_payload["scene_segments_dir"] = _resolve_path(
            processing_payload.get("scene_segments_dir"), base_dir
        )
        processing_payload["linear_rule"] = LinearSelectionConfig(
            **dict(processing_payload.get("linear_rule", {}))
        )

        shot_boundary_payload = dict(payload.get("shot_boundary", {}))
        shot_boundary_payload["output_dir"] = _resolve_path(
            shot_boundary_payload.get("output_dir"), base_dir
        )
        shot_boundary = ShotBoundaryConfig(**shot_boundary_payload)
        selector = str(processing_payload.get("selector", "uniform"))
        if shot_boundary.enabled and selector_requires_scene_boundaries(selector):
            output_dir = (
                shot_boundary.output_dir
                or processing_payload["scene_segments_dir"]
                or (data_root / "scene-segments")
            )
            configured_scene_dir = processing_payload["scene_segments_dir"]
            if configured_scene_dir is not None and configured_scene_dir != output_dir:
                raise ValueError(
                    "processing.scene_segments_dir and shot_boundary.output_dir must match"
                )
            shot_boundary_payload["output_dir"] = output_dir
            processing_payload["scene_segments_dir"] = output_dir
            shot_boundary = ShotBoundaryConfig(**shot_boundary_payload)

        processing = ProcessingConfig(**processing_payload)
        embedding = PECoreEmbeddingConfig(**dict(payload.get("embedding", {})))

        upload_payload = dict(payload.get("upload", {}))
        upload_payload["metadata_template"] = _resolve_path(
            upload_payload.get("metadata_template"), base_dir
        )
        upload = UploadConfig(**upload_payload)

        links_file = _resolve_path(
            payload.get("links_file"), base_dir, (data_root or Path("data")) / "input" / "links.txt"
        )
        metadata_root = _resolve_path(
            payload.get("metadata_root"), base_dir, (data_root or Path("data")) / "metadata"
        )
        return cls(
            data_root=data_root or Path("data"),
            links_file=links_file or Path("data/input/links.txt"),
            metadata_root=metadata_root or Path("data/metadata"),
            tools=tools,
            download=download,
            progress=progress,
            shot_boundary=shot_boundary,
            archive=archive,
            processing=processing,
            embedding=embedding,
            upload=upload,
            cleanup=CleanupConfig(**dict(payload.get("cleanup", {}))),
            minimum_free_bytes=int(payload.get("minimum_free_bytes", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config for a checkpoint without exposing secrets."""
        value = asdict(self)
        for key in ("data_root", "links_file", "metadata_root"):
            value[key] = str(value[key])
        value["processing"]["scene_segments_dir"] = (
            str(value["processing"]["scene_segments_dir"])
            if value["processing"]["scene_segments_dir"] is not None
            else None
        )
        value["shot_boundary"]["output_dir"] = (
            str(value["shot_boundary"]["output_dir"])
            if value["shot_boundary"]["output_dir"] is not None
            else None
        )
        value["upload"]["metadata_template"] = (
            str(value["upload"]["metadata_template"])
            if value["upload"]["metadata_template"] is not None
            else None
        )
        value["archive"]["video_extensions"] = list(value["archive"]["video_extensions"])
        value["processing"]["decode_checkpoints"] = list(value["processing"]["decode_checkpoints"])
        value["embedding"]["image_extensions"] = list(value["embedding"]["image_extensions"])
        return value
