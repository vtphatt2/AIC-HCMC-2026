"""Replaceable shot-boundary detection and manifest persistence.

The batch pipeline consumes detector-agnostic JSON manifests.  TransNetV2 is
the default adapter, but callers can inject another ``ShotBoundaryDetector``
without changing orchestration or keyframe selection.
"""
from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from preprocess.batch.config import ShotBoundaryConfig
from preprocess.batch.models import VideoAsset
from preprocess.batch.provenance import (
    atomic_json_write,
    file_fingerprint_matches,
    sha256_file,
)
from preprocess.keyframes.contracts import SceneSegment
from preprocess.progress import ProgressConfig, ProgressReporter, TqdmProgressReporter


def _to_builtin(value: Any) -> Any:
    """Convert numpy/torch scalar containers into JSON-friendly values."""
    item = getattr(value, "item", None)
    if callable(item):
        return _to_builtin(item())
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item_value) for key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item_value) for item_value in value]
    return value


@dataclass(frozen=True)
class ShotBoundaryDetection:
    """Detector output before it is persisted as a per-video manifest."""

    video_id: str
    backend: str
    segments: tuple[Mapping[str, Any], ...]
    threshold: float
    fps: float | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "backend": self.backend,
            "threshold": self.threshold,
            "fps": self.fps,
            "segments": [_to_builtin(segment) for segment in self.segments],
        }


@dataclass(frozen=True)
class ShotBoundaryArtifact:
    """One validated output manifest produced or reused by the stage."""

    video_id: str
    path: Path
    backend: str
    scene_count: int
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "path": str(self.path),
            "backend": self.backend,
            "scene_count": self.scene_count,
            "cached": self.cached,
        }


class ShotBoundaryDetector(ABC):
    """Dependency boundary for any video shot/scene detector."""

    name: str

    @abstractmethod
    def detect(self, asset: VideoAsset) -> ShotBoundaryDetection:
        raise NotImplementedError


class TransNetV2ShotBoundaryDetector(ShotBoundaryDetector):
    """Run the pinned ``transnetv2-pytorch`` implementation lazily."""

    name = "transnetv2"

    def __init__(self, config: ShotBoundaryConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from transnetv2_pytorch import TransNetV2
        except ImportError as exc:
            raise RuntimeError(
                "TransNetV2 is not installed in the preprocess environment; "
                "run: python -m pip install -r preprocess/requirements.txt"
            ) from exc
        self._model = TransNetV2(device=self.config.device)
        return self._model

    def detect(self, asset: VideoAsset) -> ShotBoundaryDetection:
        if not asset.path.is_file():
            raise FileNotFoundError(f"Video file not found for shot detection: {asset.path}")

        # TransNetV2 enables deterministic algorithms, while CUDA CuBLAS still
        # emits a non-fatal warning unless CUBLAS_WORKSPACE_CONFIG is exported
        # before Python starts.  We keep the behavior unchanged and suppress
        # only this repetitive warning so it does not corrupt tqdm's terminal
        # region; users who require strict reproducibility can still set the
        # documented CUDA variable in the shell.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Deterministic behavior was enabled with either.*",
                category=UserWarning,
            )
            model = self._get_model()
            fps_value = float(model.get_video_fps(str(asset.path)))
            fps = fps_value if fps_value > 0 else None
            if self.config.window_batch_size == 1:
                video_frames, single_frame_predictions, all_frame_predictions = (
                    model.predict_video(str(asset.path), quiet=True)
                )
            else:
                video_frames, single_frame_predictions, all_frame_predictions = (
                    self._predict_video_batched(
                        model,
                        asset.path,
                        self.config.window_batch_size,
                    )
                )
            scenes = model.predictions_to_scenes_with_data(
                single_frame_predictions,
                fps=fps,
                threshold=self.config.threshold,
            )
            del video_frames, single_frame_predictions, all_frame_predictions
            cleanup_memory = getattr(model, "_cleanup_memory", None)
            if callable(cleanup_memory):
                cleanup_memory()
        if not isinstance(scenes, list):
            raise RuntimeError(
                f"TransNetV2 returned an unsupported scene result for {asset.video_id}: "
                f"{type(scenes).__name__}"
            )

        segments: list[Mapping[str, Any]] = []
        for scene in scenes:
            value = _to_builtin(scene)
            if not isinstance(value, Mapping):
                raise RuntimeError(f"TransNetV2 returned an invalid scene for {asset.video_id}")
            segments.append(dict(value))

        return ShotBoundaryDetection(
            video_id=asset.video_id,
            backend=self.name,
            segments=tuple(segments),
            threshold=self.config.threshold,
            fps=fps,
        )

    @staticmethod
    def _predict_video_batched(model: Any, path: Path, batch_size: int):
        """Run the package's 100/50 frame windows in configurable batches."""
        try:
            import ffmpeg
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Batched TransNetV2 requires ffmpeg-python, numpy and torch"
            ) from exc

        stream, _ = (
            ffmpeg.input(str(path))
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(stream, np.uint8).reshape([-1, 27, 48, 3])
        frames = torch.from_numpy(np.array(video, copy=True)).to(model.device)
        if len(frames) == 0:
            raise RuntimeError(f"TransNetV2 decoded no frames: {path}")

        window_size = 100
        step_size = 50
        padding_start = 25
        remainder = len(frames) % step_size
        padding_end = 25 + step_size - (remainder if remainder else step_size)
        padded = torch.cat(
            [frames[0].unsqueeze(0)] * padding_start
            + [frames]
            + [frames[-1].unsqueeze(0)] * padding_end,
            dim=0,
        )
        windows = [
            padded[start : start + window_size]
            for start in range(0, len(padded) - window_size + 1, step_size)
        ]
        singles = []
        all_frames = []
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                batch = torch.stack(windows[start : start + batch_size], dim=0)
                single_prediction, all_prediction = model.predict_raw(batch)
                singles.append(single_prediction[:, 25:75, 0].reshape(-1).cpu())
                all_frames.append(all_prediction[:, 25:75, 0].reshape(-1).cpu())
        single = torch.cat(singles, dim=0)[: len(frames)]
        all_prediction = torch.cat(all_frames, dim=0)[: len(frames)]
        return frames, single, all_prediction


class ShotBoundaryDetectorRegistry:
    """Named registry for detector adapters."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[ShotBoundaryConfig], ShotBoundaryDetector]] = {}

    def register(self, name: str, factory: Callable[[ShotBoundaryConfig], ShotBoundaryDetector]) -> None:
        if not name.strip():
            raise ValueError("shot boundary backend name must not be empty")
        self._factories[name] = factory

    def create(self, name: str, config: ShotBoundaryConfig) -> ShotBoundaryDetector:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"Unknown shot boundary backend: {name}") from exc
        return factory(config)


def default_shot_boundary_registry() -> ShotBoundaryDetectorRegistry:
    registry = ShotBoundaryDetectorRegistry()
    registry.register("transnetv2", TransNetV2ShotBoundaryDetector)
    registry.register("transnet", TransNetV2ShotBoundaryDetector)
    return registry


def _parse_seconds(value: object, *, path: Path) -> int:
    try:
        return round(float(value) * 1000)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scene timestamp in {path}: {value!r}") from exc


def _manifest_records(payload: object, path: Path) -> tuple[Mapping[str, Any], ...]:
    records: object = payload.get("segments") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list):
        raise ValueError(f"Scene-boundary manifest must contain a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError(f"Scene boundary must be an object: {path}")
        result.append(item)
    return tuple(result)


def load_scene_segments(path: Path) -> list[SceneSegment]:
    """Load detector-agnostic scene records as half-open millisecond intervals."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = _manifest_records(payload, path)
    manifest_fps: float | None = None
    if isinstance(payload, Mapping) and payload.get("fps") not in (None, ""):
        try:
            manifest_fps = float(payload["fps"])
        except (TypeError, ValueError):
            manifest_fps = None

    segments: list[SceneSegment] = []
    for item in records:
        if "start_ms" in item and "end_ms" in item:
            start_ms = int(item["start_ms"])
            end_ms = int(item["end_ms"])
        elif "start_time" in item and "end_time" in item:
            start_ms = _parse_seconds(item["start_time"], path=path)
            end_ms = _parse_seconds(item["end_time"], path=path)
        elif (
            manifest_fps
            and manifest_fps > 0
            and "start_frame" in item
            and "end_frame" in item
        ):
            start_ms = round(int(item["start_frame"]) / manifest_fps * 1000)
            end_ms = round((int(item["end_frame"]) + 1) / manifest_fps * 1000)
        else:
            raise ValueError(
                f"Scene boundary needs start_ms/end_ms, start_time/end_time, "
                f"or start_frame/end_frame with fps: {path}"
            )

        # TransNetV2 timestamps are rounded to milliseconds.  Use frame data to
        # recover a non-empty interval for very short scenes when available.
        if end_ms <= start_ms and manifest_fps and "start_frame" in item and "end_frame" in item:
            end_ms = round((int(item["end_frame"]) + 1) / manifest_fps * 1000)
        segments.append(SceneSegment(start_ms=start_ms, end_ms=end_ms))
    return segments


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[SceneSegment]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid scene-boundary manifest {path}: {exc}") from exc
    segments = load_scene_segments(path)
    metadata = dict(payload) if isinstance(payload, Mapping) else {"segments": payload}
    return metadata, segments


def write_scene_boundary_manifest(
    path: Path,
    detection: ShotBoundaryDetection,
    *,
    source: VideoAsset | None = None,
) -> None:
    """Atomically persist one detector result so a killed SSH session is safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = detection.to_manifest()
    if source is not None:
        stat = source.path.stat()
        payload["source"] = {
            "path": source.path.name,
            "fingerprint": {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "sha256": sha256_file(source.path),
            },
        }
    atomic_json_write(path, payload)


class ShotBoundaryPipeline:
    """Run, validate and cache detector manifests for a set of videos."""

    def __init__(
        self,
        detector: ShotBoundaryDetector,
        output_dir: Path,
        *,
        overwrite: bool = False,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.detector = detector
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.progress = progress or TqdmProgressReporter(ProgressConfig())

    def run(self, assets: Sequence[VideoAsset]) -> tuple[ShotBoundaryArtifact, ...]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[ShotBoundaryArtifact] = []
        for asset in self.progress.iterate(
            assets,
            total=len(assets),
            desc="shot boundaries",
            unit="video",
        ):
            output_path = self.output_dir / f"{asset.video_id}.json"
            cached = False
            payload: dict[str, Any] | None = None
            segments: list[SceneSegment] = []
            if output_path.is_file() and not self.overwrite:
                try:
                    candidate, candidate_segments = _read_manifest(output_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    candidate = None
                    candidate_segments = []
                if candidate is not None and self._cache_matches(candidate, asset):
                    payload = candidate
                    segments = candidate_segments
                    cached = bool(segments)

            if not cached:
                detection = self.detector.detect(asset)
                if detection.video_id != asset.video_id:
                    raise ValueError(
                        f"Shot detector returned video_id {detection.video_id!r} "
                        f"for {asset.video_id!r}"
                    )
                write_scene_boundary_manifest(output_path, detection, source=asset)
                _, segments = _read_manifest(output_path)
                backend = detection.backend
            else:
                backend = str(payload.get("backend", self.detector.name))
            if not segments:
                raise ValueError(f"Shot-boundary manifest has no scenes: {output_path}")
            artifacts.append(
                ShotBoundaryArtifact(
                    video_id=asset.video_id,
                    path=output_path,
                    backend=backend,
                    scene_count=len(segments),
                    cached=cached,
                )
            )
        return tuple(artifacts)

    def _cache_matches(self, payload: Mapping[str, Any], asset: VideoAsset) -> bool:
        if str(payload.get("backend", "")) != self.detector.name:
            return False
        expected_threshold = getattr(getattr(self.detector, "config", None), "threshold", None)
        if expected_threshold is not None:
            try:
                if float(payload.get("threshold")) != float(expected_threshold):
                    return False
            except (TypeError, ValueError):
                return False

        source = payload.get("source")
        if not isinstance(source, Mapping):
            # Manifests from the first implementation have no source
            # fingerprint.  Recompute them once so a changed source cannot be
            # mistaken for a valid cache.
            return False
        fingerprint = source.get("fingerprint")
        if not isinstance(fingerprint, Mapping):
            return False
        return file_fingerprint_matches(asset.path, fingerprint)
