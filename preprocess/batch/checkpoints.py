"""Atomic state/checkpoint persistence for resumable SSH runs."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from preprocess.batch.models import BatchState, utc_now


class CheckpointStore:
    """Persist one JSON state document without exposing half-written files."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"state": BatchState.NEW.value, "events": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint must be a JSON object: {self.path}")
        payload.setdefault("events", [])
        payload.setdefault("state", BatchState.NEW.value)
        return payload

    def write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp", delete=False
        ) as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            temporary_path = Path(handle.name)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self.path)

    def transition(
        self,
        state: BatchState,
        *,
        payload: Mapping[str, Any] | None = None,
        allowed_from: set[BatchState] | None = None,
    ) -> dict[str, Any]:
        current = self.load()
        current_state = BatchState(current.get("state", BatchState.NEW.value))
        if allowed_from is not None and current_state not in allowed_from:
            raise RuntimeError(f"Invalid checkpoint transition {current_state.value} -> {state.value}")
        event = {"state": state.value, "at": utc_now(), "payload": dict(payload or {})}
        events = list(current.get("events", []))
        events.append(event)
        current["state"] = state.value
        current["updated_at"] = event["at"]
        current["events"] = events
        if payload:
            current.update(payload)
        self.write(current)
        return current
