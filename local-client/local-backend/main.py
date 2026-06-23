"""
AIC 2026 — Local Backend (Development Playground)
Run with: uvicorn main:app --reload --port 8000
"""
import os
import time
import importlib
import inspect
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

from app.data_provider import DataProvider, sample_subdir
from app.strategies.base_strategy import BaseStrategy, FETCH_CAP

STRATEGIES_DIR = Path(__file__).parent / "app" / "strategies"
SAMPLE_KEYFRAMES_DIR = sample_subdir("keyframes")
_strategies: dict[str, BaseStrategy] = {}
_data_provider: DataProvider | None = None


def discover_strategies(data_provider: DataProvider) -> dict[str, BaseStrategy]:
    """
    Scan strategies/ for .py files. Any class that inherits BaseStrategy
    (except BaseStrategy itself) is instantiated and registered under the
    file's stem as the strategy_id.
    """
    found = {}
    for path in sorted(STRATEGIES_DIR.glob("*.py")):
        if path.stem.startswith("_") or path.stem == "base_strategy":
            continue
        module_name = f"app.strategies.{path.stem}"
        try:
            module = importlib.import_module(module_name)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseStrategy) and cls is not BaseStrategy:
                    instance = cls(data_provider)
                    found[path.stem] = instance
                    print(f"  OK {path.stem}  [{cls.name}]  by {cls.author}")
                    break  # one strategy class per file
        except Exception as exc:
            print(f"  ERR {path.stem}: {exc}")
    return found


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _strategies, _data_provider
    print("Starting local backend…")
    _data_provider = DataProvider()
    print("Discovering strategies…")
    _strategies = discover_strategies(_data_provider)
    print(f"Ready — {len(_strategies)} strategy/strategies available.\n")
    yield


app = FastAPI(
    title="AIC 2026 Local Backend",
    description="Development playground for retrieval strategies.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if SAMPLE_KEYFRAMES_DIR.is_dir():
    app.mount(
        "/static/frames",
        StaticFiles(directory=str(SAMPLE_KEYFRAMES_DIR)),
        name="sample-keyframes",
    )
elif os.getenv("ENV_MODE", "MOCK").upper() == "LOCAL":
    _remote_base = os.getenv("REMOTE_SERVER_URL", "").rstrip("/")
    _http_client = httpx.AsyncClient(timeout=30)

    @app.get("/static/frames/{path:path}")
    async def proxy_frame(path: str):
        url = f"{_remote_base}/static/frames/{path}"
        resp = await _http_client.get(url, headers={"ngrok-skip-browser-warning": "1"})
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"Remote server returned {resp.status_code}")
        return Response(
            content=resp.content,
            media_type=resp.headers.get("content-type", "image/jpeg"),
        )


# ── Request / Response models ─────────────────────────────────────────────────

class QueryGroup(BaseModel):
    semantic_query: str = ""
    text_query: str = ""
    temporal_offset_ms: int = 0   # ms after the previous group's result window


class SearchRequest(BaseModel):
    strategy_id: str
    query_groups: list[QueryGroup]
    top_k: int = 100


class TranscriptSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class TranscriptResultItem(BaseModel):
    video_id: str
    youtube_id: str = ""
    start_time_ms: int
    end_time_ms: int
    text: str
    score: float
    nearest_frame_id: str | None = None
    nearest_timestamp_ms: int | None = None
    frame_image_url: str | None = None
    # New optional fields for Vietnamese upgrade
    normalized_query: str = ""
    match_type: str = "token_overlap"
    window_text: str = ""
    window_start_time_ms: int | None = None
    window_end_time_ms: int | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":     "ok",
        "env_mode":   os.getenv("ENV_MODE", "MOCK"),
        "strategies": len(_strategies),
        "sample_keyframes_dir": str(SAMPLE_KEYFRAMES_DIR),
        "sample_static_mounted": SAMPLE_KEYFRAMES_DIR.is_dir(),
    }


@app.get("/api/static-debug")
async def static_debug(path: str = "L01_V001/000022.jpg"):
    target = (SAMPLE_KEYFRAMES_DIR / path).resolve()
    return {
        "sample_keyframes_dir": str(SAMPLE_KEYFRAMES_DIR.resolve()),
        "path": path,
        "target": str(target),
        "exists": target.exists(),
        "is_file": target.is_file(),
        "size": target.stat().st_size if target.exists() else None,
    }


@app.get("/api/strategies")
async def list_strategies():
    """Return all discovered strategies for the frontend dropdown."""
    return [
        {
            "id":          sid,
            "name":        s.name,
            "description": s.description,
            "author":      s.author,
            "version":     s.version,
        }
        for sid, s in _strategies.items()
    ]


@app.post("/api/search")
async def search(req: SearchRequest):
    print(req)
    """Run a search with the selected strategy and return ranked results."""
    if req.strategy_id not in _strategies:
        raise HTTPException(404, f"Strategy '{req.strategy_id}' not found. Available: {list(_strategies)}")

    strategy = _strategies[req.strategy_id]
    t0 = time.monotonic()
    top_k = min(max(req.top_k, 1), FETCH_CAP)

    try:
        results = await strategy.search(
            [g.model_dump() for g in req.query_groups], 
            limit=top_k,
        )
    except TimeoutError as exc:
        raise HTTPException(408, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Strategy error: {exc}")

    results = results[:top_k]

    return {
        "results":           results,
        "strategy_id":       req.strategy_id,
        "total":             len(results),
        "execution_time_ms": int((time.monotonic() - t0) * 1000),
    }


@app.post("/api/search-transcript")
async def search_transcript(req: TranscriptSearchRequest):
    """Search transcripts independently, without running a full strategy pipeline."""
    if not _data_provider:
        raise HTTPException(503, "DataProvider is not ready.")
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty.")

    t0 = time.monotonic()
    top_k = min(max(req.top_k, 1), 200)

    try:
        results = await _data_provider.search_transcripts(req.query.strip(), limit=top_k)
    except Exception as exc:
        raise HTTPException(500, f"Transcript search error: {exc}")

    return {
        "results":           results[:top_k],
        "total":             min(len(results), top_k),
        "execution_time_ms": int((time.monotonic() - t0) * 1000),
    }
