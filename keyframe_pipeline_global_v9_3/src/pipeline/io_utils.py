"""Small filesystem/id helpers shared by every phase."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def safe_video_id(entry_name: str) -> str:
    path = Path(entry_name)
    parent = "__".join(path.parts[:-1])
    raw = f"{parent}__{path.stem}" if parent else path.stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_") or "video"
