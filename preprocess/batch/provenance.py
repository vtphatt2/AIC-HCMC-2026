"""Content fingerprints and runtime provenance for resumable artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


PROVENANCE_FILE_NAME = "provenance.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Describe one regular file using a portable path and content digest."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot fingerprint missing file: {path}")
    record: dict[str, Any] = {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path)
        ),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return record


def inventory(
    root: Path,
    *,
    include: Iterable[Path] | None = None,
    exclude_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic content inventory below ``root``."""
    exclude_names = exclude_names or set()
    paths = (
        sorted(path for path in root.rglob("*") if path.is_file() and path.name not in exclude_names)
        if include is None
        else sorted(path for path in include if path.is_file() and path.name not in exclude_names)
    )
    return [file_record(path, relative_to=root) for path in paths]


def digest_records(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash canonical file records, independent of host absolute paths."""
    canonical = [dict(record) for record in records]
    canonical.sort(key=lambda record: str(record.get("path", "")))
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def digest_directory(
    root: Path,
    *,
    include: Iterable[Path] | None = None,
    exclude_names: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(digest, inventory)`` for a directory payload."""
    records = inventory(root, include=include, exclude_names=exclude_names)
    return digest_records(records), records


def atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON to avoid partial provenance files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def git_revision(start: Path) -> str | None:
    """Return the current repository revision when available."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def package_versions() -> dict[str, str]:
    """Return versions of packages that can affect generated artifacts."""
    distributions = (
        "numpy",
        "Pillow",
        "tqdm",
        "kaggle",
        "transnetv2-pytorch",
        "open_clip_torch",
        "torch",
        "torchvision",
        "huggingface_hub",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def tool_versions(tools: Mapping[str, str] | None) -> dict[str, str]:
    """Record short executable version strings without failing a completed run."""
    if not tools:
        return {}
    versions: dict[str, str] = {}
    for name, executable in sorted(tools.items()):
        if not executable or executable.startswith("python:"):
            continue
        resolved = shutil.which(executable) or executable
        try:
            completed = subprocess.run(
                [resolved, "--version"],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            versions[name] = f"unavailable: {exc}"
            continue
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        versions[name] = output.splitlines()[0][:500] if output else f"exit:{completed.returncode}"
    return versions


def runtime_provenance(
    *,
    root: Path,
    tools: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect runtime facts that can affect generated artifacts."""
    return {
        "python": sys.version,
        "platform": sys.platform,
        "git_revision": git_revision(root),
        "packages": package_versions(),
        "tools": tool_versions(tools),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_mode": os.environ.get("PREPROCESS_DETERMINISTIC", "0") == "1",
    }


def apply_reproducibility_policy(*, mode: str, seed: int | None, cublas_workspace_config: str) -> None:
    """Configure process-level deterministic settings before ML libraries load."""
    if mode != "strict":
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    os.environ["PREPROCESS_DETERMINISTIC"] = "1"
    if seed is None:
        return
    os.environ["PREPROCESS_SEED"] = str(seed)
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
