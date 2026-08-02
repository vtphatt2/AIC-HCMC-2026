"""Downloader adapters; the default implementation invokes aria2c safely."""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from preprocess.batch.config import DownloadConfig
from preprocess.batch.models import ArchiveInput, DownloadResult


class ArchiveDownloader(ABC):
    """Replaceable boundary for direct archive download implementations."""

    @abstractmethod
    def download(self, request: ArchiveInput, destination_dir: Path) -> DownloadResult:
        raise NotImplementedError


class Aria2ArchiveDownloader(ArchiveDownloader):
    """Download one direct HTTP(S) archive with resume support."""

    def __init__(
        self,
        executable: str = "aria2c",
        config: DownloadConfig | None = None,
        *,
        show_progress: bool = True,
    ) -> None:
        self.executable = executable
        self.config = config or DownloadConfig()
        self.show_progress = show_progress

    def build_command(self, request: ArchiveInput, destination_dir: Path) -> list[str]:
        command = [
            self.executable,
            "--dir",
            str(destination_dir),
            "--out",
            request.archive_name,
            "--continue=true" if self.config.continue_download else "--continue=false",
            "--auto-file-renaming=true" if self.config.auto_file_renaming else "--auto-file-renaming=false",
            "--max-tries",
            str(self.config.max_tries),
            "--retry-wait",
            str(self.config.retry_wait_seconds),
            "--timeout",
            str(self.config.timeout_seconds),
            "--connect-timeout",
            str(self.config.connect_timeout_seconds),
            "--split",
            str(self.config.split_count),
            "--max-connection-per-server",
            str(self.config.max_concurrent_connections),
        ]
        if self.config.check_integrity:
            command.append("--check-integrity=true")
        command.append(request.url)
        return command

    def download(self, request: ArchiveInput, destination_dir: Path) -> DownloadResult:
        destination_dir.mkdir(parents=True, exist_ok=True)
        target = destination_dir / request.archive_name
        resumed = target.is_file() or target.with_name(f"{target.name}.aria2").is_file()
        command = self.build_command(request, destination_dir)
        completed = subprocess.run(
            command,
            text=True,
            capture_output=not self.show_progress,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"aria2c failed for {request.url} (exit {completed.returncode}): {details[-2000:]}"
            )
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"aria2c reported success but archive is missing/empty: {target}")
        control_file = target.with_name(f"{target.name}.aria2")
        if control_file.exists():
            raise RuntimeError(f"aria2c left an incomplete control file: {control_file}")
        return DownloadResult(
            request=request,
            path=target,
            command=tuple(command),
            resumed=resumed,
            size_bytes=target.stat().st_size,
        )
