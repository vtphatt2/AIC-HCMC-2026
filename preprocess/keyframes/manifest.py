"""Versioned JSON manifests for reproducible selection and rendering."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from preprocess.keyframes.contracts import RenderProfile, SelectedFrame, VideoInfo, VideoSource

SELECTION_SCHEMA_VERSION = 1
RENDERED_SCHEMA_VERSION = 1


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete JSON file atomically, avoiding half-written manifests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def source_fingerprint(source: VideoSource) -> dict[str, int]:
    """Cheap change detector; callers may add a content hash in future."""
    stat = source.path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_selection_manifest(
    path: Path,
    source: VideoSource,
    video: VideoInfo,
    selector_name: str,
    selector_version: str,
    selector_config: dict[str, Any],
    selected: Sequence[SelectedFrame],
) -> None:
    _atomic_json_write(
        path,
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "video_id": source.video_id,
            "source": {"path": str(source.path), "fingerprint": source_fingerprint(source)},
            "video": asdict(video),
            "selector": {
                "name": selector_name,
                "version": selector_version,
                "config": selector_config,
            },
            "selected_frames": [asdict(frame) for frame in selected],
        },
    )


def write_rendered_manifest(
    path: Path,
    source: VideoSource,
    profile: RenderProfile,
    selection_manifest_path: Path,
    frames: list[dict[str, Any]],
) -> None:
    _atomic_json_write(
        path,
        {
            "schema_version": RENDERED_SCHEMA_VERSION,
            "video_id": source.video_id,
            "source": {"path": str(source.path), "fingerprint": source_fingerprint(source)},
            "profile": asdict(profile),
            # The selection manifest is the provenance source for score/reason
            # data; rendering a new profile never needs to invoke a selector.
            "selection_manifest_path": str(selection_manifest_path),
            "frames": frames,
        },
    )
