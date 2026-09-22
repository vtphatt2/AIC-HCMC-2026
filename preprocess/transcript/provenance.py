"""Small provenance helpers owned by the retained legacy transcript tools."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Describe one source artifact using a portable path and content digest."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot fingerprint missing file: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def digest_records(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash canonical records independently of absolute host paths."""
    canonical = sorted((dict(record) for record in records), key=lambda item: str(item.get("path", "")))
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON atomically so interrupted index builds leave no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
