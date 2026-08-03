"""Environment and disk checks performed before any download."""
from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

from preprocess.batch.config import BatchConfig, selector_requires_scene_boundaries


@dataclass(frozen=True)
class PreflightResult:
    tools: dict[str, str]
    free_bytes: int
    root: Path


class PreflightChecker:
    """Check configured external tools and available filesystem capacity."""

    def __init__(self, config: BatchConfig) -> None:
        self.config = config

    def run(self) -> PreflightResult:
        tool_paths: dict[str, str] = {}
        for name, executable in (
            ("aria2c", self.config.tools.aria2c),
            ("ffmpeg", self.config.tools.ffmpeg),
            ("ffprobe", self.config.tools.ffprobe),
        ):
            resolved = shutil.which(executable)
            if resolved is None:
                raise RuntimeError(f"Required executable not found: {executable}")
            tool_paths[name] = resolved

        if self.config.upload.enabled:
            resolved = shutil.which(self.config.tools.kaggle)
            if resolved is None:
                raise RuntimeError(f"Kaggle executable not found: {self.config.tools.kaggle}")
            tool_paths["kaggle"] = resolved

        if (
            self.config.shot_boundary.enabled
            and selector_requires_scene_boundaries(self.config.processing.selector)
            and self.config.shot_boundary.backend in {"transnetv2", "transnet"}
        ):
            if importlib.util.find_spec("transnetv2_pytorch") is None:
                raise RuntimeError(
                    "TransNetV2 Python package is not installed in the preprocess environment"
                )
            tool_paths["transnetv2"] = "python:transnetv2_pytorch"

        root = self.config.data_root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(root).free
        if free_bytes < self.config.minimum_free_bytes:
            raise RuntimeError(
                f"Insufficient free disk space: {free_bytes} < {self.config.minimum_free_bytes} bytes"
            )
        return PreflightResult(tools=tool_paths, free_bytes=free_bytes, root=Path(root))
