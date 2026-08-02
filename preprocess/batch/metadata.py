"""External metadata access; the provider never mutates the source metadata."""
from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class MetadataProvider(ABC):
    """Read metadata independently from downloaded archives."""

    @abstractmethod
    def path_for(self, video_id: str) -> Path:
        raise NotImplementedError

    def load(self, video_id: str) -> Mapping[str, Any]:
        path = self.path_for(video_id)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata not found for {video_id}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Metadata must be a JSON object: {path}")
        return payload

    def copy_to(self, video_id: str, destination_root: Path) -> Path:
        source = self.path_for(video_id)
        if not source.is_file():
            raise FileNotFoundError(f"Metadata not found for {video_id}: {source}")
        destination = destination_root / f"{video_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination


class JsonMetadataProvider(MetadataProvider):
    """A provider backed by ``<metadata_root>/<video_id>.json`` files."""

    def __init__(self, metadata_root: Path) -> None:
        self.metadata_root = metadata_root.expanduser()

    def path_for(self, video_id: str) -> Path:
        if not video_id or Path(video_id).name != video_id:
            raise ValueError(f"Unsafe video_id for metadata lookup: {video_id!r}")
        return self.metadata_root / f"{video_id}.json"
