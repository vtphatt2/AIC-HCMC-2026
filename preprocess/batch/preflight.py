"""Environment and disk checks performed before any download."""
from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from preprocess.batch.config import BatchConfig, selector_requires_scene_boundaries


@dataclass(frozen=True)
class PreflightResult:
    tools: dict[str, str]
    versions: dict[str, str]
    packages: dict[str, str]
    free_bytes: int
    root: Path


class PreflightChecker:
    """Check configured external tools and available filesystem capacity."""

    def __init__(self, config: BatchConfig) -> None:
        self.config = config

    def run(self) -> PreflightResult:
        tool_paths: dict[str, str] = {}
        versions: dict[str, str] = {}
        packages: dict[str, str] = {}
        for name, executable in (
            ("aria2c", self.config.tools.aria2c),
            ("ffmpeg", self.config.tools.ffmpeg),
            ("ffprobe", self.config.tools.ffprobe),
        ):
            resolved = shutil.which(executable)
            if resolved is None:
                raise RuntimeError(f"Required executable not found: {executable}")
            tool_paths[name] = resolved
            versions[name] = self._version(resolved)

        self._check_ffmpeg_showinfo(tool_paths["ffmpeg"])

        if self.config.upload.enabled:
            resolved = shutil.which(self.config.tools.kaggle)
            if resolved is None:
                raise RuntimeError(f"Kaggle executable not found: {self.config.tools.kaggle}")
            tool_paths["kaggle"] = resolved
            versions["kaggle"] = self._version(resolved)
            self._check_kaggle_configuration(resolved)
            if self.config.upload.mode == "version" and self.config.upload.dataset_ref:
                self._check_kaggle_status(resolved, self.config.upload.dataset_ref)

            metadata_template = self.config.upload.metadata_template
            if metadata_template is None or not metadata_template.is_file():
                raise RuntimeError(
                    "upload.metadata_template must point to an existing dataset-metadata.json"
                )
            try:
                metadata = json.loads(metadata_template.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid Kaggle dataset metadata: {metadata_template}") from exc
            if not isinstance(metadata, dict):
                raise RuntimeError("Kaggle dataset metadata must be a JSON object")

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
            packages["transnetv2-pytorch"] = self._package_version("transnetv2-pytorch")

        if self.config.embedding.enabled:
            for module in ("torch", "open_clip", "PIL"):
                if importlib.util.find_spec(module) is None:
                    raise RuntimeError(
                        f"Embedding dependency is not installed in the preprocess environment: {module}"
                    )
            for distribution in ("torch", "torchvision", "open_clip_torch", "Pillow", "numpy"):
                packages[distribution] = self._package_version(distribution)

        for distribution in ("tqdm", "kaggle"):
            if distribution == "kaggle" and not self.config.upload.enabled:
                continue
            packages[distribution] = self._package_version(distribution)

        root = self.config.data_root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(root).free
        if free_bytes < self.config.minimum_free_bytes:
            raise RuntimeError(
                f"Insufficient free disk space: {free_bytes} < {self.config.minimum_free_bytes} bytes"
            )
        return PreflightResult(
            tools=tool_paths,
            versions=versions,
            packages=packages,
            free_bytes=free_bytes,
            root=Path(root),
        )

    @staticmethod
    def _version(executable: str) -> str:
        try:
            completed = subprocess.run(
                [executable, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable: {exc}"
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return output.splitlines()[0][:500] if output else f"exit:{completed.returncode}"

    @staticmethod
    def _package_version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required Python distribution is not installed: {distribution}"
            ) from exc

    @staticmethod
    def _check_kaggle_configuration(executable: str) -> None:
        try:
            completed = subprocess.run(
                [executable, "config", "view"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Unable to inspect Kaggle CLI configuration: {exc}") from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "Kaggle CLI authentication/configuration is unavailable: "
                f"{(completed.stderr or completed.stdout or '').strip()[-1000:]}"
            )

    @staticmethod
    def _check_kaggle_status(executable: str, dataset_ref: str) -> None:
        completed = subprocess.run(
            [executable, "datasets", "status", dataset_ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Configured Kaggle dataset is not accessible: {dataset_ref}: "
                f"{(completed.stderr or completed.stdout or '').strip()[-1000:]}"
            )

    @staticmethod
    def _check_ffmpeg_showinfo(executable: str) -> None:
        """Ensure the decoder timeline filter used by keyframe scanning exists."""
        try:
            completed = subprocess.run(
                [executable, "-hide_banner", "-h", "filter=showinfo"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Unable to inspect FFmpeg showinfo filter: {exc}") from exc
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0 or "Filter showinfo" not in output or "checksum" not in output:
            raise RuntimeError(
                "FFmpeg must provide the showinfo filter with the checksum option "
                "for frame timeline scanning"
            )
