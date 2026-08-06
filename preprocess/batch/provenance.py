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
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping


PROVENANCE_FILE_NAME = "provenance.json"


class FileDigestCache:
    """Cache content digests while a file's stat identity remains unchanged.

    The cache is process-local by design.  It avoids repeatedly streaming large
    videos and staging payloads during one run without trusting a path alone:
    replacement, resize, timestamp changes and inode changes all produce a new
    key.  A before/after stat check prevents publishing a digest for a file that
    changed while it was being read.
    """

    def __init__(self, *, minimum_cache_bytes: int = 1024 * 1024) -> None:
        self._digests: dict[tuple[str, int, int, int, int, int], str] = {}
        self._lock = threading.Lock()
        self.minimum_cache_bytes = minimum_cache_bytes

    @staticmethod
    def _identity(path: Path, stat: os.stat_result) -> tuple[str, int, int, int, int, int]:
        return (
            str(path.resolve()),
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )

    def digest(
        self,
        path: Path,
        *,
        chunk_size: int = 1024 * 1024,
        cache_result: bool = False,
    ) -> str:
        path = Path(path)
        before = path.stat()
        identity = self._identity(path, before)
        with self._lock:
            cached = (
                self._digests.get(identity)
                if cache_result or before.st_size >= self.minimum_cache_bytes
                else None
            )
        if cached is not None:
            return cached

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        after = path.stat()
        if self._identity(path, after) != identity:
            raise RuntimeError(f"File changed while hashing: {path}")

        value = digest.hexdigest()
        if cache_result or after.st_size >= self.minimum_cache_bytes:
            with self._lock:
                self._digests[identity] = value
        return value

    def clear(self) -> None:
        """Clear cached values; primarily useful for bounded test lifetimes."""
        with self._lock:
            self._digests.clear()


_FILE_DIGEST_CACHE = FileDigestCache()


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
    cache_result: bool = False,
) -> str:
    """Return a cached SHA256 digest without loading the file into memory."""
    return _FILE_DIGEST_CACHE.digest(
        path,
        chunk_size=chunk_size,
        cache_result=cache_result,
    )


def file_fingerprint_matches(
    path: Path,
    fingerprint: Mapping[str, Any],
    *,
    full_audit: bool = False,
) -> bool:
    """Validate a file using stat first and SHA256 only when necessary.

    New manifests carry both mtime and a content hash.  On the same filesystem,
    equal size and mtime are a cheap resume check.  If the timestamp changed
    (for example after copying artifacts to another host), the content digest
    remains authoritative.  ``full_audit`` always verifies content when a hash
    is available.
    """
    try:
        stat = path.stat()
        if int(fingerprint.get("size_bytes")) != stat.st_size:
            return False
        expected_hash = fingerprint.get("sha256")
        expected_mtime = fingerprint.get("mtime_ns")
        expected_ctime = fingerprint.get("ctime_ns")
        if not full_audit and expected_mtime is not None:
            if int(expected_mtime) == stat.st_mtime_ns and (
                expected_ctime is None or int(expected_ctime) == stat.st_ctime_ns
            ):
                return True
        if expected_hash is not None:
            return str(expected_hash) == sha256_file(path)
        return expected_mtime is not None and int(expected_mtime) == stat.st_mtime_ns
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


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


def package_versions(distributions: Iterable[str] | None = None) -> dict[str, str]:
    """Return versions of packages that can affect generated artifacts."""
    selected = tuple(distributions) if distributions is not None else (
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
    for distribution in selected:
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
