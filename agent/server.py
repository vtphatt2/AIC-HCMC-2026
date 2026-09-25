"""Small optional HTTP controller; manual VORTA does not import this module."""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .codex import CodexPlanner
from .search_agent import SearchAgent
from .state import SearchJobManager, VerifyJobManager
from .tools import VortaClient
from .verify_agent import CodexVerifier, VerifyAgent


VORTA_BACKEND_URL = os.getenv("VORTA_BACKEND_URL", "http://127.0.0.1:8000")


def _run_search(query: str, *, top_k: int, cancel_event, on_status):
    client = VortaClient(VORTA_BACKEND_URL)
    try:
        return SearchAgent(CodexPlanner(), client).run(
            query, top_k=top_k, cancel_event=cancel_event, on_status=on_status
        )
    finally:
        client.close()


manager = SearchJobManager(_run_search)


def _run_verify(query: str, candidate: dict, cancel_event):
    client = VortaClient(VORTA_BACKEND_URL, timeout_seconds=8)
    try:
        return VerifyAgent(CodexVerifier(), client).run(query, candidate, cancel_event=cancel_event)
    finally:
        client.close()


verify_manager = VerifyJobManager(_run_verify)
app = FastAPI(title="VORTA Search Agent", version="0.1")


class StartSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=20, ge=1, le=30)


class StartVerifyRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    candidate: dict


@app.get("/api/agent/search")
def get_search():
    return manager.snapshot()


@app.post("/api/agent/search")
def start_search(request: StartSearchRequest):
    try:
        before = manager.snapshot()["query_id"]
        state = manager.start(request.query, request.top_k)
        if state["query_id"] != before:
            verify_manager.reset(state["query"])
        return state
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/agent/search/cancel")
def cancel_search():
    return manager.cancel()


@app.delete("/api/agent/search")
def reset_search():
    state = manager.reset()
    verify_manager.reset()
    return state


@app.get("/api/agent/verify")
def get_verify():
    return verify_manager.snapshot()


@app.post("/api/agent/verify")
def start_verify(request: StartVerifyRequest):
    official_query = manager.snapshot()["query"]
    if official_query and request.query.strip() != official_query:
        raise HTTPException(409, "Candidate query differs from the current official query")
    try:
        return verify_manager.start(request.query, request.candidate)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/agent/verify/cancel")
def cancel_verify():
    return verify_manager.cancel()


@app.delete("/api/agent/verify")
def reset_verify():
    return verify_manager.reset()
