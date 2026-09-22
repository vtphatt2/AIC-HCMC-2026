"""ZIP-native preprocessing facade and result-contract validation.

The heavy GPU implementation lives in ``keyframe_pipeline_global_v9_3``.  This
module is the stable public interface owned by :mod:`preprocess`: it selects one
source, enables inter-stage streaming, and validates the final challenge ZIP.
It deliberately uses only the Python standard library so ``--dry-run`` and
``inspect`` work before the GPU environment is installed.
"""
from __future__ import annotations

import ast
import json
import struct
import subprocess
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


MEDIA_SUFFIXES = {
    ".avi", ".bmp", ".gif", ".jpeg", ".jpg", ".m4v", ".mkv", ".mov",
    ".mp4", ".png", ".webm", ".webp",
}


def _source_data_id(source_url: str | None, source_zip: Path | None) -> str:
    if source_url:
        name = PurePosixPath(urllib.parse.unquote(urllib.parse.urlsplit(source_url).path)).name
    elif source_zip is not None:
        name = source_zip.name
    else:  # guarded by ZipPipelineConfig, kept defensive for direct calls
        raise ValueError("a ZIP source is required")
    stem = Path(name or "input.zip").stem
    # Organizer archives historically used ``Videos_L26_a.zip`` and now use
    # range names such as ``Video_N001-N010.zip``. Normalize either transport
    # prefix so work/output names describe the data range, not its container.
    return stem.removeprefix("Videos_").removeprefix("Video_") or "input"


@dataclass(frozen=True)
class ZipPipelineConfig:
    """Configuration for one downloaded-or-local ZIP preprocessing run."""

    source_url: str | None = None
    source_zip: Path | None = None
    work_root: Path = Path("data/zip-preprocess")
    archive: Path | None = None
    profile: str = "balanced"
    device: str = "cuda"
    batch_size: int = 64
    prefetch_batches: int = 3
    transnet_batch_size: int = 16
    transnet_decode_workers: int = 4
    transnet_prefetch_windows: int = 256
    keyframe_strategy: str = "tiered"
    keyframes_per_second: float = 0.3
    min_keyframes_per_scene: int = 1
    max_keyframes_per_scene: int = 20
    limit: int | None = None
    parallel_stages: bool = True
    delete_source_before_package: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if (self.source_url is None) == (self.source_zip is None):
            raise ValueError("provide exactly one of source_url or source_zip")
        if self.source_url is not None and not self.source_url.strip():
            raise ValueError("source_url cannot be empty")
        for name in (
            "batch_size", "prefetch_batches", "transnet_batch_size",
            "transnet_decode_workers", "transnet_prefetch_windows",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.keyframe_strategy not in {"tiered", "linear"}:
            raise ValueError("keyframe_strategy must be tiered or linear")
        if self.keyframes_per_second <= 0:
            raise ValueError("keyframes_per_second must be positive")
        if self.min_keyframes_per_scene < 1:
            raise ValueError("min_keyframes_per_scene must be positive")
        if self.max_keyframes_per_scene < 0:
            raise ValueError("max_keyframes_per_scene must be non-negative")
        if self.max_keyframes_per_scene and self.max_keyframes_per_scene < self.min_keyframes_per_scene:
            raise ValueError("max_keyframes_per_scene must be >= min_keyframes_per_scene")

    @property
    def data_id(self) -> str:
        return _source_data_id(self.source_url, self.source_zip)

    @property
    def result_archive(self) -> Path:
        if self.archive is not None:
            return self.archive
        return self.work_root / f"{self.data_id}_results.zip"


def build_engine_command(
    config: ZipPipelineConfig,
    *,
    repository_root: Path | None = None,
) -> list[str]:
    """Build the argument-safe command for the ZIP-native GPU engine."""

    root = repository_root or Path(__file__).resolve().parent.parent
    command = [
        "bash",
        str(root / "keyframe_pipeline_global_v9_3" / "run_pipeline.sh"),
    ]
    if config.source_url is not None:
        command.extend(("--url", config.source_url))
    else:
        command.extend(("--zip", str(config.source_zip)))
    command.extend((
        "--work-root", str(config.work_root),
        "--archive", str(config.result_archive),
        "--profile", config.profile,
        "--device", config.device,
        "--batch-size", str(config.batch_size),
        "--prefetch-batches", str(config.prefetch_batches),
        "--transnet-batch-size", str(config.transnet_batch_size),
        "--transnet-decode-workers", str(config.transnet_decode_workers),
        "--transnet-prefetch-windows", str(config.transnet_prefetch_windows),
        "--keyframe-strategy", config.keyframe_strategy,
        "--keyframes-per-second", str(config.keyframes_per_second),
        "--min-keyframes-per-scene", str(config.min_keyframes_per_scene),
        "--max-keyframes-per-scene", str(config.max_keyframes_per_scene),
        "--parallel-stages" if config.parallel_stages else "--sequential-stages",
    ))
    if config.limit is not None:
        command.extend(("--limit", str(config.limit)))
    if config.delete_source_before_package:
        command.append("--delete-source-before-package")
    if config.local_files_only:
        command.append("--local-files-only")
    return command


@dataclass(frozen=True)
class ResultInspection:
    archive: Path
    format_version: int
    video_count: int
    keyframe_count: int
    embedding_dim: int | None

    def to_dict(self) -> dict:
        return {
            "archive": str(self.archive),
            "format_version": self.format_version,
            "video_count": self.video_count,
            "keyframe_count": self.keyframe_count,
            "embedding_dim": self.embedding_dim,
        }


def _read_npy_shape(payload: bytes) -> tuple[int, ...]:
    stream = memoryview(payload)
    if len(stream) < 10 or bytes(stream[:6]) != b"\x93NUMPY":
        raise ValueError("embedding is not a valid NPY file")
    major = stream[6]
    if major == 1:
        header_size = struct.unpack("<H", stream[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        if len(stream) < 12:
            raise ValueError("truncated NPY header")
        header_size = struct.unpack("<I", stream[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"unsupported NPY version: {major}")
    header_end = header_start + header_size
    if header_end > len(stream):
        raise ValueError("truncated NPY header")
    encoding = "utf-8" if major == 3 else "latin1"
    header = ast.literal_eval(bytes(stream[header_start:header_end]).decode(encoding).strip())
    shape = header.get("shape") if isinstance(header, dict) else None
    if not isinstance(shape, tuple) or not all(isinstance(value, int) and value >= 0 for value in shape):
        raise ValueError("invalid NPY shape")
    return shape


class ResultArchiveValidator:
    """Validate the compact challenge archive without extracting it."""

    def validate(self, archive: Path) -> ResultInspection:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as source:
            bad_member = source.testzip()
            if bad_member is not None:
                raise ValueError(f"corrupt ZIP member: {bad_member}")
            names = set(source.namelist())
            media = sorted(name for name in names if PurePosixPath(name).suffix.lower() in MEDIA_SUFFIXES)
            if media:
                raise ValueError(f"result ZIP contains media artifact: {media[0]}")
            if "manifest.json" not in names:
                raise ValueError("result ZIP is missing manifest.json")
            manifest = json.loads(source.read("manifest.json"))
            format_version = manifest.get("format_version")
            if format_version not in (2, 3):
                raise ValueError("unsupported manifest format_version (expected 2 or 3)")
            videos = manifest.get("videos")
            if not isinstance(videos, list) or manifest.get("num_videos") != len(videos):
                raise ValueError("manifest video count is inconsistent")

            total_keyframes = 0
            embedding_dim: int | None = None
            for video in videos:
                if not isinstance(video, dict):
                    raise ValueError("manifest videos[] entries must be objects")
                paths = [video.get(field) for field in ("scenes", "keyframes", "embeddings")]
                if not all(isinstance(path, str) and path in names for path in paths):
                    raise ValueError(f"incomplete artifacts for video: {video.get('video_id')}")
                keyframes = json.loads(source.read(video["keyframes"]))
                count = keyframes.get("num_keyframes")
                rows = keyframes.get("keyframes")
                if not isinstance(count, int) or not isinstance(rows, list) or count != len(rows):
                    raise ValueError(f"invalid keyframe metadata for video: {video.get('video_id')}")
                if video.get("num_keyframes") != count:
                    raise ValueError(f"manifest keyframe count mismatch: {video.get('video_id')}")
                shape = _read_npy_shape(source.read(video["embeddings"]))
                if len(shape) != 2 or shape[0] != count:
                    raise ValueError(f"embedding shape mismatch for video: {video.get('video_id')}")
                if embedding_dim is None:
                    embedding_dim = shape[1]
                elif shape[1] != embedding_dim:
                    raise ValueError("embedding dimensions are inconsistent")
                total_keyframes += count

        return ResultInspection(archive, format_version, len(videos), total_keyframes, embedding_dim)


def run_pipeline(
    config: ZipPipelineConfig,
    *,
    repository_root: Path | None = None,
    runner=subprocess.run,
) -> ResultInspection:
    """Run the engine and validate its final archive before reporting success."""

    command = build_engine_command(config, repository_root=repository_root)
    completed = runner(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ZIP preprocessing failed with exit code {completed.returncode}")
    inspection = ResultArchiveValidator().validate(config.result_archive)
    if inspection.format_version != 3:
        raise ValueError("new pipeline output must use manifest format_version 3")
    return inspection


def shell_join(command: Sequence[str]) -> str:
    """Return a display-only safely quoted command."""

    import shlex

    return shlex.join(command)
