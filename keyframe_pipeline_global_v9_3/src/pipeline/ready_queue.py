"""Filesystem-backed hand-off from TransNet producers to embedding consumers."""
from __future__ import annotations

from pathlib import Path

from .io_utils import safe_video_id


class ReadyEntryQueue:
    """Yield videos as soon as their atomic TransNet artifacts are available."""

    def __init__(self, entries: list, out_dir: Path, producer_done: Path) -> None:
        self._pending = list(entries)
        self._out_dir = out_dir
        self._producer_done = producer_done

    def take_ready(self) -> list:
        ready = []
        waiting = []
        for entry in self._pending:
            video_out = self._out_dir / safe_video_id(entry.name)
            if (video_out / "scenes.json").is_file() and (video_out / "keyframes.json").is_file():
                ready.append(entry)
            else:
                waiting.append(entry)
        self._pending = waiting
        return ready

    @property
    def producer_finished(self) -> bool:
        return self._producer_done.is_file()

    def take_missing(self) -> list:
        missing, self._pending = self._pending, []
        return missing

    def __bool__(self) -> bool:
        return bool(self._pending)
