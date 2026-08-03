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

    def stage_is_complete(self, name: str, fingerprint: str) -> bool:
        """Return true only when a stage marker matches this run's inputs."""
        stage = self.load().get("stages", {}).get(name, {})
        return stage.get("status") == "completed" and stage.get("fingerprint") == fingerprint

    def start_stage(self, name: str, *, fingerprint: str) -> dict[str, Any]:
        """Atomically mark a stage as running before invoking external work."""
        current = self.load()
        stages = dict(current.get("stages", {}))
        previous = stages.get(name, {})
        attempt = int(previous.get("attempt", 0)) + 1
        timestamp = utc_now()
        stages[name] = {
            "status": "running",
            "fingerprint": fingerprint,
            "attempt": attempt,
            "started_at": timestamp,
        }
        current["stages"] = stages
        current["current_stage"] = name
        self._append_stage_event(current, name, "started", timestamp)
        self.write(current)
        return current

    def complete_stage(
        self,
        name: str,
        *,
        fingerprint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically publish a successful stage marker after its artifacts exist."""
        current = self.load()
        stages = dict(current.get("stages", {}))
        previous = dict(stages.get(name, {}))
        timestamp = utc_now()
        previous.update(
            {
                "status": "completed",
                "fingerprint": fingerprint,
                "completed_at": timestamp,
            }
        )
        if payload:
            previous["payload"] = dict(payload)
        stages[name] = previous
        current["stages"] = stages
        if current.get("current_stage") == name:
            current["current_stage"] = None
        self._append_stage_event(current, name, "completed", timestamp, payload=payload)
        self.write(current)
        return current

    def invalidate_stage(self, name: str, *, reason: str) -> dict[str, Any]:
        """Mark a stale completion so the next attempt executes the stage again."""
        current = self.load()
        stages = dict(current.get("stages", {}))
        previous = dict(stages.get(name, {}))
        timestamp = utc_now()
        previous.update({"status": "stale", "invalidated_at": timestamp, "reason": reason})
        stages[name] = previous
        current["stages"] = stages
        self._append_stage_event(current, name, "invalidated", timestamp, payload={"reason": reason})
        self.write(current)
        return current

    @staticmethod
    def _append_stage_event(
        current: dict[str, Any],
        name: str,
        status: str,
        timestamp: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        events = list(current.get("events", []))
        event: dict[str, Any] = {
            "state": current.get("state", BatchState.NEW.value),
            "stage": name,
            "status": status,
            "at": timestamp,
        }
        if payload:
            event["payload"] = dict(payload)
        events.append(event)
        current["events"] = events
        current["updated_at"] = timestamp

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
