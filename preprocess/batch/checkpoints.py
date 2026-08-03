"""Atomic state/checkpoint persistence for resumable SSH runs."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
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
        previous = dict(stages.get(name, {}))
        attempt = int(previous.get("attempt", 0)) + 1
        timestamp = utc_now()
        stage = {
            "status": "running",
            "fingerprint": fingerprint,
            "attempt": attempt,
            "started_at": timestamp,
        }
        # Preserve successful per-video checkpoints when retrying the same
        # stage inputs. A changed stage fingerprint invalidates the old set.
        if previous.get("fingerprint") == fingerprint and isinstance(previous.get("videos"), Mapping):
            stage["videos"] = dict(previous["videos"])
        stages[name] = stage
        current["stages"] = stages
        current["current_stage"] = name
        current["current_video"] = None
        self._append_stage_event(current, name, "started", timestamp)
        self.write(current)
        return current

    def video_is_complete(self, stage_name: str, video_id: str, fingerprint: str) -> bool:
        """Return true when one video checkpoint matches the current inputs."""
        stage = self.load().get("stages", {}).get(stage_name, {})
        videos = stage.get("videos", {})
        video = videos.get(video_id, {}) if isinstance(videos, Mapping) else {}
        return video.get("status") == "completed" and video.get("fingerprint") == fingerprint

    def video_payload(self, stage_name: str, video_id: str) -> dict[str, Any] | None:
        """Return the serialized result attached to a completed video."""
        stage = self.load().get("stages", {}).get(stage_name, {})
        videos = stage.get("videos", {})
        video = videos.get(video_id, {}) if isinstance(videos, Mapping) else {}
        payload = video.get("payload")
        return dict(payload) if isinstance(payload, Mapping) else None

    def start_video(self, stage_name: str, video_id: str, *, fingerprint: str) -> dict[str, Any]:
        """Atomically mark one video as running inside a stage."""
        current = self.load()
        stages = dict(current.get("stages", {}))
        stage = dict(stages.get(stage_name, {}))
        raw_videos = stage.get("videos", {})
        videos = dict(raw_videos) if isinstance(raw_videos, Mapping) else {}
        previous = dict(videos.get(video_id, {}))
        attempt = int(previous.get("attempt", 0)) + 1
        timestamp = utc_now()
        videos[video_id] = {
            "status": "running",
            "fingerprint": fingerprint,
            "attempt": attempt,
            "started_at": timestamp,
        }
        stage["videos"] = videos
        stages[stage_name] = stage
        current["stages"] = stages
        current["current_stage"] = stage_name
        current["current_video"] = video_id
        self._append_stage_event(
            current,
            stage_name,
            "started",
            timestamp,
            video_id=video_id,
        )
        self.write(current)
        return current

    def complete_video(
        self,
        stage_name: str,
        video_id: str,
        *,
        fingerprint: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically publish one video's validated artifacts and result."""
        current = self.load()
        stages = dict(current.get("stages", {}))
        stage = dict(stages.get(stage_name, {}))
        raw_videos = stage.get("videos", {})
        videos = dict(raw_videos) if isinstance(raw_videos, Mapping) else {}
        previous = dict(videos.get(video_id, {}))
        timestamp = utc_now()
        elapsed_seconds = self._elapsed_seconds(previous.get("started_at"), timestamp)
        record: dict[str, Any] = {
            **previous,
            "status": "completed",
            "fingerprint": fingerprint,
            "completed_at": timestamp,
            "payload": dict(payload),
        }
        if elapsed_seconds is not None:
            record["elapsed_seconds"] = elapsed_seconds
        videos[video_id] = record
        stage["videos"] = videos
        stages[stage_name] = stage
        current["stages"] = stages
        if current.get("current_stage") == stage_name and current.get("current_video") == video_id:
            current["current_video"] = None
        event_payload: dict[str, Any] = {"video_id": video_id, "result": dict(payload)}
        if elapsed_seconds is not None:
            event_payload["elapsed_seconds"] = elapsed_seconds
        self._append_stage_event(
            current,
            stage_name,
            "completed",
            timestamp,
            payload=event_payload,
            video_id=video_id,
        )
        self.write(current)
        return current

    def invalidate_video(self, stage_name: str, video_id: str, *, reason: str) -> dict[str, Any]:
        """Mark one stale video checkpoint so the next retry processes it."""
        current = self.load()
        stages = dict(current.get("stages", {}))
        stage = dict(stages.get(stage_name, {}))
        raw_videos = stage.get("videos", {})
        videos = dict(raw_videos) if isinstance(raw_videos, Mapping) else {}
        previous = dict(videos.get(video_id, {}))
        timestamp = utc_now()
        previous.update({"status": "stale", "invalidated_at": timestamp, "reason": reason})
        videos[video_id] = previous
        stage["videos"] = videos
        stages[stage_name] = stage
        current["stages"] = stages
        self._append_stage_event(
            current,
            stage_name,
            "invalidated",
            timestamp,
            payload={"reason": reason},
            video_id=video_id,
        )
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
        elapsed_seconds = self._elapsed_seconds(previous.get("started_at"), timestamp)
        previous.update(
            {
                "status": "completed",
                "fingerprint": fingerprint,
                "completed_at": timestamp,
            }
        )
        if elapsed_seconds is not None:
            previous["elapsed_seconds"] = elapsed_seconds
        if payload:
            previous["payload"] = dict(payload)
        stages[name] = previous
        current["stages"] = stages
        if current.get("current_stage") == name:
            current["current_stage"] = None
            current["current_video"] = None
        event_payload = dict(payload or {})
        if elapsed_seconds is not None:
            event_payload["elapsed_seconds"] = elapsed_seconds
        self._append_stage_event(
            current,
            name,
            "completed",
            timestamp,
            payload=event_payload or None,
        )
        self.write(current)
        return current

    @staticmethod
    def _elapsed_seconds(started_at: object, completed_at: str) -> float | None:
        if not isinstance(started_at, str):
            return None
        try:
            elapsed = datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
        except ValueError:
            return None
        return round(max(0.0, elapsed.total_seconds()), 3)

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
        video_id: str | None = None,
    ) -> None:
        events = list(current.get("events", []))
        event: dict[str, Any] = {
            "state": current.get("state", BatchState.NEW.value),
            "stage": name,
            "status": status,
            "at": timestamp,
        }
        if video_id is not None:
            event["video_id"] = video_id
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
