"""Environment and disk checks performed before any download."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from preprocess.batch.config import BatchConfig


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

        root = self.config.data_root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(root).free
        if free_bytes < self.config.minimum_free_bytes:
            raise RuntimeError(
                f"Insufficient free disk space: {free_bytes} < {self.config.minimum_free_bytes} bytes"
            )
        return PreflightResult(tools=tool_paths, free_bytes=free_bytes, root=Path(root))
