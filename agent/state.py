"""One shared Search Agent job for the current official query."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from threading import Event, Lock
from uuid import uuid4


class SearchJobManager:
    def __init__(self, run_search):
        self._run_search = run_search
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vorta-search-agent")
        self._lock = Lock()
        self._generation = 0
        self._cancel_event: Event | None = None
        self._state = self._empty_state()

    def _empty_state(self):
        return {
            "query_id": None,
            "generation": self._generation,
            "query": "",
            "status": "idle",
            "plan_summary": "",
            "plan": None,
            "results": [],
            "total": 0,
            "timing_ms": None,
            "thread_id": None,
            "retrieval": None,
            "error": None,
        }

    def snapshot(self):
        with self._lock:
            return deepcopy(self._state)

    def start(self, query: str, top_k: int = 20):
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        top_k = min(max(top_k, 1), 30)
        with self._lock:
            if self._state["query"] == query and self._state["status"] in {"planning", "searching", "done"}:
                return deepcopy(self._state)
            self._cancel_current()
            self._generation += 1
            generation = self._generation
            cancel_event = Event()
            self._cancel_event = cancel_event
            self._state = self._empty_state()
            self._state.update({
                "query_id": uuid4().hex,
                "generation": generation,
                "query": query,
                "status": "planning",
            })
            self._executor.submit(self._run, generation, query, top_k, cancel_event)
            return deepcopy(self._state)

    def cancel(self):
        with self._lock:
            self._cancel_current()
            self._generation += 1
            self._state.update({
                "generation": self._generation,
                "status": "cancelled" if self._state["query_id"] else "idle",
                "plan_summary": "",
                "plan": None,
                "results": [],
                "total": 0,
                "timing_ms": None,
                "thread_id": None,
                "retrieval": None,
                "error": None,
            })
            return deepcopy(self._state)

    def reset(self):
        with self._lock:
            self._cancel_current()
            self._generation += 1
            self._state = self._empty_state()
            return deepcopy(self._state)

    def _cancel_current(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._cancel_event = None

    def _set_status(self, generation: int, status: str):
        with self._lock:
            if generation == self._generation:
                self._state["status"] = status

    def _run(self, generation: int, query: str, top_k: int, cancel_event: Event):
        if cancel_event.is_set():
            return
        try:
            result = self._run_search(
                query,
                top_k=top_k,
                cancel_event=cancel_event,
                on_status=lambda status: self._set_status(generation, status),
            )
        except InterruptedError:
            return
        except Exception as exc:
            with self._lock:
                if generation == self._generation and not cancel_event.is_set():
                    self._state.update({"status": "error", "error": str(exc)[:400]})
            return
        with self._lock:
            if generation != self._generation or cancel_event.is_set():
                return
            self._state.update({
                "status": "done",
                "plan_summary": result.get("plan_summary", ""),
                "plan": result.get("plan"),
                "results": result.get("results", []),
                "total": result.get("total", 0),
                "timing_ms": result.get("timing_ms"),
                "thread_id": result.get("thread_id"),
                "retrieval": result.get("retrieval"),
                "error": None,
            })


class VerifyJobManager:
    """One serialized, human-triggered queue shared by both browser operators."""

    def __init__(self, run_verify):
        self._run_verify = run_verify
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vorta-verify-agent")
        self._lock = Lock()
        self._generation = 0
        self._cancel_event = Event()
        self._state = self._empty_state("")

    def _empty_state(self, query):
        return {"query": query, "generation": self._generation, "status": "idle",
                "current_candidate": None, "queue": [], "results": {}, "error": None}

    def snapshot(self):
        with self._lock:
            return deepcopy(self._state)

    def _reset_locked(self, query):
        self._cancel_event.set()
        self._cancel_event = Event()
        self._generation += 1
        self._state = self._empty_state(query)

    def reset(self, query=""):
        with self._lock:
            self._reset_locked(query.strip())
            return deepcopy(self._state)

    def cancel(self):
        with self._lock:
            query = self._state["query"]
            self._reset_locked(query)
            self._state["status"] = "cancelled"
            return deepcopy(self._state)

    def start(self, query: str, candidate: dict):
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not isinstance(candidate, dict) or not candidate.get("video_id"):
            raise ValueError("candidate needs a video_id")
        steps = candidate.get("steps") or []
        if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
            raise ValueError("candidate steps must be a list of frames")
        identity = {"video_id": candidate["video_id"], "frame_id": candidate.get("frame_id"),
                    "chunk_id": candidate.get("chunk_id"),
                    "steps": [step.get("frame_id") for step in steps[:4]]}
        candidate_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        item = {"candidate_id": candidate_id, "video_id": candidate["video_id"],
                "frame_id": candidate.get("frame_id"), "chunk_id": candidate.get("chunk_id")}
        with self._lock:
            if self._state["query"] != query:
                self._reset_locked(query)
            if (self._state["current_candidate"] == item
                    or item in self._state["queue"]):
                return deepcopy(self._state)
            if candidate_id in self._state["results"]:
                if self._state["results"][candidate_id]["status"] == "done":
                    return deepcopy(self._state)
                del self._state["results"][candidate_id]
            if len(self._state["queue"]) >= 6:
                raise ValueError("Verify queue is full; wait for current candidates")
            self._state["queue"].append(item)
            self._state["status"] = "verifying"
            self._state["error"] = None
            self._executor.submit(self._run, self._generation, query, candidate, item, self._cancel_event)
            return deepcopy(self._state)

    def _run(self, generation, query, candidate, item, cancel_event):
        with self._lock:
            if generation != self._generation or cancel_event.is_set():
                return
            self._state["queue"] = [queued for queued in self._state["queue"] if queued != item]
            self._state["current_candidate"] = item
        try:
            result = self._run_verify(query, candidate, cancel_event)
            result.update({"query": query, "candidate": item, "candidate_id": item["candidate_id"], "status": "done"})
        except InterruptedError:
            with self._lock:
                if generation == self._generation:
                    self._state["current_candidate"] = None
                    self._state["status"] = "cancelled"
            return
        except Exception as exc:
            result = {"query": query, "candidate": item, "candidate_id": item["candidate_id"],
                      "status": "error", "error": str(exc)[:400]}
        with self._lock:
            if generation != self._generation or cancel_event.is_set():
                return
            self._state["current_candidate"] = None
            self._state["results"][item["candidate_id"]] = result
            self._state["status"] = "verifying" if self._state["queue"] else result["status"]
            self._state["error"] = result.get("error")
