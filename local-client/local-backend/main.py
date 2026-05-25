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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.data_provider import DataProvider
from app.strategies.base_strategy import BaseStrategy

STRATEGIES_DIR = Path(__file__).parent / "app" / "strategies"
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
                    print(f"  ✓ {path.stem}  [{cls.name}]  by {cls.author}")
                    break  # one strategy class per file
        except Exception as exc:
            print(f"  ✗ {path.stem}: {exc}")
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


# ── Request / Response models ─────────────────────────────────────────────────

class QueryGroup(BaseModel):
    semantic_query: str = ""
    text_query: str = ""
    temporal_offset_ms: int = 0   # ms after the previous group's result window


class SearchRequest(BaseModel):
    strategy_id: str
    query_groups: list[QueryGroup]
    top_k: int = 100


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":     "ok",
        "env_mode":   os.getenv("ENV_MODE", "MOCK"),
        "strategies": len(_strategies),
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
    """Run a search with the selected strategy and return ranked results."""
    if req.strategy_id not in _strategies:
        raise HTTPException(404, f"Strategy '{req.strategy_id}' not found. Available: {list(_strategies)}")

    strategy = _strategies[req.strategy_id]
    t0 = time.monotonic()

    try:
        results = await strategy.search([g.model_dump() for g in req.query_groups])
    except TimeoutError as exc:
        raise HTTPException(408, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Strategy error: {exc}")

    results = results[: req.top_k]

    return {
        "results":           results,
        "strategy_id":       req.strategy_id,
        "total":             len(results),
        "execution_time_ms": int((time.monotonic() - t0) * 1000),
    }
