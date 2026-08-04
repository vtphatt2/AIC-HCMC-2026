"""Small filesystem locks for mutually exclusive lot-level operations."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


class ExclusiveFileLock:
    """Hold an advisory, non-blocking lock for the lifetime of a context."""

    def __init__(self, path: Path, *, purpose: str) -> None:
        self.path = path
        self.purpose = purpose
        self._handle = None

    def __enter__(self) -> "ExclusiveFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                f"Cannot acquire {self.purpose} lock; another process may be using {self.path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
