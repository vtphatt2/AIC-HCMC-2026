"""Discover source videos while preserving their original filenames."""
from __future__ import annotations

from pathlib import Path

from preprocess.batch.models import VideoAsset


class VideoDiscovery:
    """Map ``L21_V030.mp4`` to ``video_id == 'L21_V030'`` without renaming it."""

    def __init__(self, video_extensions: tuple[str, ...] | list[str] | None = None) -> None:
        self.video_extensions = tuple(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in (video_extensions or (".mp4", ".mkv", ".mov", ".avi", ".webm"))
        )

    def discover(self, source_root: Path, lot_id: str) -> list[VideoAsset]:
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        assets: list[VideoAsset] = []
        seen_ids: set[str] = set()
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.video_extensions:
                continue
            video_id = path.stem
            if not video_id or video_id in seen_ids:
                raise ValueError(f"Duplicate/empty video_id discovered in {source_root}: {video_id!r}")
            seen_ids.add(video_id)
            assets.append(
                VideoAsset(
                    video_id=video_id,
                    path=path,
                    lot_id=lot_id,
                    source_name=path.name,
                )
            )
        if not assets:
            raise ValueError(f"No supported video files discovered in {source_root}")
        return assets
