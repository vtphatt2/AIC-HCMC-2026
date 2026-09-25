import time
from threading import Event

from .tools import VortaClient, VortaToolError, search_vorta


class SearchAgent:
    def __init__(self, planner, vorta: VortaClient):
        self.planner = planner
        self.vorta = vorta

    def run(self, query: str, *, top_k: int = 20, cancel_event: Event | None = None,
            on_status=None) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Search Agent cancelled")
        started = time.monotonic()
        capabilities = self.vorta.capabilities()
        plan, thread_id = self.planner.plan(query, capabilities, cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Search Agent cancelled")
        planning_ms = int((time.monotonic() - started) * 1000)
        if on_status is not None:
            on_status("searching")

        attempts = [plan.primary]
        if plan.fallback is not None:
            attempts.append(plan.fallback)
        tool_errors = []
        response = None
        attempts_run = 0
        retrieval_started = time.monotonic()
        for index, attempt in enumerate(attempts[:2]):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Search Agent cancelled")
            attempts_run += 1
            try:
                response = search_vorta(self.vorta, attempt, top_k=top_k)
            except VortaToolError as exc:
                tool_errors.append(str(exc))
                if index + 1 >= len(attempts):
                    raise
                continue
            if response.get("results") or index + 1 >= len(attempts):
                break
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Search Agent cancelled")

        retrieval_ms = int((time.monotonic() - retrieval_started) * 1000)
        response = response or {"results": [], "total": 0}
        return {
            "query": query,
            "thread_id": thread_id,
            "status": "done",
            "plan_summary": plan.plan_summary,
            "plan": plan.model_dump(),
            "attempts": attempts_run,
            "results": response.get("results", []),
            "total": response.get("total", len(response.get("results", []))),
            "retrieval": {
                key: value for key, value in response.items() if key != "results"
            },
            "timing_ms": {
                "planning": planning_ms,
                "retrieval": retrieval_ms,
                "total": int((time.monotonic() - started) * 1000),
            },
            "tool_errors": tool_errors,
        }
