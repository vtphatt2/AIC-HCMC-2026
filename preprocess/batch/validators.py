"""Composable validation rules for archives, videos and generated artifacts."""
from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from preprocess.batch.config import ProcessingConfig, ToolConfig
from preprocess.batch.metadata import MetadataProvider
from preprocess.batch.models import ProcessingResult, ValidationReport, VideoAsset
from preprocess.keyframes.contracts import RenderProfile


@dataclass(frozen=True)
class VideoValidationContext:
    asset: VideoAsset
    metadata: Mapping[str, Any]
    processing: ProcessingResult


class ValidationRule(ABC):
    """One independently testable validation rule."""

    @abstractmethod
    def apply(self, context: VideoValidationContext, report: ValidationReport) -> None:
        raise NotImplementedError


class MetadataRule(ValidationRule):
    """Validate external metadata without requiring an FPS field."""

    def apply(self, context: VideoValidationContext, report: ValidationReport) -> None:
        if not isinstance(context.metadata, Mapping):
            report.add_error("metadata.invalid", "Metadata is not a JSON object")
            return
        report.measurements["metadata_fps"] = context.metadata.get("fps")
        if context.metadata.get("fps") in (None, ""):
            report.add_warning(
                "metadata.fps_missing",
                "Metadata has no FPS; timing will come from the media probe/PTS",
            )


class VideoProbeRule(ValidationRule):
    """Validate the measured video information and decode checkpoints."""

    def __init__(self, executable: str, checkpoints: Sequence[float]) -> None:
        self.executable = executable
        self.checkpoints = tuple(checkpoints)

    def apply(self, context: VideoValidationContext, report: ValidationReport) -> None:
        path = context.asset.path
        if not path.is_file():
            report.add_error("video.missing", f"Video file is missing: {path}")
            return
        if path.stat().st_size <= 0:
            report.add_error("video.empty", f"Video file is empty: {path}")
            return

        info = dict(context.processing.video_info)
        report.measurements.update(
            {
                "video_size_bytes": path.stat().st_size,
                "duration_ms": info.get("duration_ms"),
                "fps": info.get("fps"),
                "width": info.get("width"),
                "height": info.get("height"),
                "frame_count": info.get("frame_count"),
                "codec": info.get("codec"),
                "timing_source": "media_probe_pts",
            }
        )
        if int(info.get("duration_ms") or 0) <= 0:
            report.add_error("video.duration", "Video duration is missing or non-positive")
        if int(info.get("width") or 0) <= 0 or int(info.get("height") or 0) <= 0:
            report.add_error("video.dimensions", "Video dimensions are missing or non-positive")
        if info.get("fps") is None:
            report.add_warning("video.fps_unavailable", "Media probe did not expose a constant FPS; PTS will be authoritative")

        duration_seconds = max(float(info.get("duration_ms") or 0) / 1000, 0.0)
        for checkpoint in self.checkpoints:
            timestamp = min(duration_seconds, duration_seconds * checkpoint)
            if checkpoint >= 1.0 and duration_seconds > 0.001:
                timestamp = duration_seconds - 0.001
            self._decode_checkpoint(path, timestamp, checkpoint, report)

    def _decode_checkpoint(
        self,
        path: Path,
        timestamp_seconds: float,
        checkpoint: float,
        report: ValidationReport,
    ) -> None:
        command = [
            self.executable,
            "-v",
            "error",
            "-i",
            str(path),
            "-ss",
            f"{timestamp_seconds:.6f}",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            report.add_error(
                "video.decode_checkpoint",
                f"Failed to decode checkpoint at {checkpoint:.3f}: {(completed.stderr or '').strip()[-1000:]}",
                checkpoint=checkpoint,
                timestamp_seconds=timestamp_seconds,
            )


class ArtifactRule(ValidationRule):
    """Validate selection/render manifests and all rendered image files."""

    def __init__(self, profile: RenderProfile) -> None:
        self.profile = profile

    def apply(self, context: VideoValidationContext, report: ValidationReport) -> None:
        selection = self._load_json(context.processing.selection_manifest_path, report, "selection")
        rendered = self._load_json(context.processing.rendered_manifest_path, report, "rendered")
        if selection is None or rendered is None:
            return

        selected_frames = selection.get("selected_frames")
        rendered_frames = rendered.get("frames")
        if not isinstance(selected_frames, list) or not isinstance(rendered_frames, list):
            report.add_error("artifacts.manifest_shape", "Keyframe manifests have invalid frame lists")
            return
        report.measurements["selected_keyframes"] = len(selected_frames)
        report.measurements["rendered_keyframes"] = len(rendered_frames)
        if not selected_frames:
            report.add_error("artifacts.empty_selection", "Selector produced no keyframes")
        if len(selected_frames) != len(rendered_frames):
            report.add_error("artifacts.count_mismatch", "Selection and rendered frame counts differ")

        selected_numbers = [self._source_frame_number(item) for item in selected_frames]
        rendered_numbers = [self._source_frame_number(item) for item in rendered_frames]
        if selected_numbers != sorted(set(selected_numbers)):
            report.add_error("artifacts.selection_order", "Selected frame numbers are not unique and ordered")
        if selected_numbers != rendered_numbers:
            report.add_error("artifacts.frame_mapping", "Rendered frames do not match selected frames")

        for frame in rendered_frames:
            self._validate_rendered_frame(frame, context.processing.rendered_manifest_path, report)

        source_fingerprint = selection.get("source", {}).get("fingerprint", {})
        if source_fingerprint:
            current_stat = context.asset.path.stat()
            if source_fingerprint.get("size_bytes") != current_stat.st_size:
                report.add_error("artifacts.source_changed", "Source video size differs from selection fingerprint")
            if source_fingerprint.get("mtime_ns") != current_stat.st_mtime_ns:
                report.add_warning("artifacts.source_mtime_changed", "Source video mtime differs from selection fingerprint")

    @staticmethod
    def _load_json(path: Path, report: ValidationReport, label: str) -> dict[str, Any] | None:
        if not path.is_file():
            report.add_error(f"artifacts.{label}_missing", f"Missing {label} manifest: {path}")
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.add_error(f"artifacts.{label}_invalid", f"Invalid {label} manifest: {exc}")
            return None
        if not isinstance(payload, dict):
            report.add_error(f"artifacts.{label}_shape", f"{label} manifest must be a JSON object")
            return None
        return payload

    @staticmethod
    def _source_frame_number(item: Mapping[str, Any]) -> int:
        frame = item.get("ref") or item.get("frame") or item
        return int(frame["source_frame_number"])

    def _validate_rendered_frame(
        self,
        frame: Mapping[str, Any],
        manifest_path: Path,
        report: ValidationReport,
    ) -> None:
        path_value = frame.get("path")
        if not path_value:
            report.add_error("artifacts.frame_path", "Rendered frame has no path")
            return
        path = Path(str(path_value))
        candidates = [path]
        if not path.is_absolute():
            candidates.append(manifest_path.parent / path.name)
        actual = next((candidate for candidate in candidates if candidate.is_file()), None)
        if actual is None:
            report.add_error("artifacts.frame_missing", f"Rendered frame is missing: {path}")
            return
        try:
            from PIL import Image

            with Image.open(actual) as image:
                image.verify()
                width, height = image.size
        except Exception as exc:  # Pillow exposes multiple format-specific errors.
            report.add_error("artifacts.frame_invalid", f"Cannot read rendered frame {actual}: {exc}")
            return
        expected_width = int(frame.get("width") or 0)
        expected_height = int(frame.get("height") or 0)
        if (width, height) != (expected_width, expected_height):
            report.add_error(
                "artifacts.frame_dimensions",
                f"Rendered frame dimensions differ for {actual}: {(width, height)} != {(expected_width, expected_height)}",
            )
        if self.profile.image_format == "jpeg" and actual.suffix.lower() not in {".jpg", ".jpeg"}:
            report.add_error("artifacts.frame_extension", f"Expected JPEG output: {actual}")
        if self.profile.image_format == "png" and actual.suffix.lower() != ".png":
            report.add_error("artifacts.frame_extension", f"Expected PNG output: {actual}")


class VideoValidationPipeline:
    """Run an injected collection of rules and return a report."""

    def __init__(self, rules: Sequence[ValidationRule]) -> None:
        self.rules = tuple(rules)

    def validate(self, context: VideoValidationContext) -> ValidationReport:
        report = ValidationReport(subject=context.asset.video_id)
        for rule in self.rules:
            try:
                rule.apply(context, report)
            except Exception as exc:
                report.add_error("validator.exception", f"{type(rule).__name__} failed: {exc}")
        return report.finish()


def default_video_validation_pipeline(
    *,
    tools: ToolConfig,
    processing_config: ProcessingConfig,
) -> VideoValidationPipeline:
    return VideoValidationPipeline(
        (
            MetadataRule(),
            VideoProbeRule(tools.ffmpeg, processing_config.decode_checkpoints),
            ArtifactRule(processing_config.render_profile()),
        )
    )


def load_video_metadata(provider: MetadataProvider, video_id: str) -> Mapping[str, Any]:
    """Small injectable helper used by the orchestrator and tests."""
    return provider.load(video_id)
