import os
import time
import importlib
import inspect
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

from app.data_provider import DataProvider
from app.strategies.base_strategy import BaseStrategy, FETCH_CAP
from app.db import postgres_client, milvus_client

STRATEGIES_DIR = Path(__file__).parent / "app" / "strategies"
_strategies: dict[str, BaseStrategy] = {}
_data_provider: DataProvider | None = None
logger = logging.getLogger(__name__)


def discover_strategies(data_provider: DataProvider) -> dict[str, BaseStrategy]:
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
                    print(f"  ✓ {path.stem}  ({cls.name}  by {cls.author})")
                    break
        except Exception as exc:
            print(f"  ✗ {path.stem}: {exc}")
    return found


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _strategies, _data_provider
    print("Initialising DB schema...")
    await postgres_client.init_schema()
    milvus_client.connect()
    milvus_client.create_collection_if_missing()

    print("Loading DataProvider...")
    _data_provider = DataProvider()

    print("Discovering strategies...")
    _strategies = discover_strategies(_data_provider)
    print(f"Server ready — {len(_strategies)} strategy/strategies loaded.")

    yield

    await postgres_client.close_pool()


app = FastAPI(title="AIC 2026 Remote Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve pre-extracted frame images
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Models ────────────────────────────────────────────────────────────────────

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
    return {"status": "ok", "env_mode": os.getenv("ENV_MODE"), "strategies": len(_strategies)}


@app.get("/api/static-debug")
async def static_debug(path: str = "frames/L01_V001/000022.jpg"):
    target = (STATIC_DIR / path).resolve()
    return {
        "static_dir": str(STATIC_DIR.resolve()),
        "path": path,
        "target": str(target),
        "exists": target.exists(),
        "is_file": target.is_file(),
        "size": target.stat().st_size if target.exists() else None,
    }


@app.get("/api/warmup_text_encoder")
@app.post("/api/warmup_text_encoder")
async def warmup_text_encoder():
    if _data_provider is None:
        raise HTTPException(503, "DataProvider is not ready.")

    t0 = time.monotonic()
    try:
        result = await asyncio.to_thread(
            _data_provider.warmup_text_encoder,
            "warmup query",
        )
    except Exception as exc:
        logger.info(
            "[TIMER] warmup_text_encoder %.3f ms status=error error=%s",
            (time.monotonic() - t0) * 1000,
            exc,
        )
        raise HTTPException(500, f"Text encoder warmup error: {exc}")

    logger.info(
        "[TIMER] warmup_text_encoder %.3f ms status=ok model_load_ms=%.3f encode_ms=%.3f device=%s",
        (time.monotonic() - t0) * 1000,
        float(result["model_load_ms"]),
        float(result["encode_ms"]),
        result["device"],
    )
    return result


@app.get("/api/strategies")
async def list_strategies():
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
    t0 = time.monotonic()
    top_k = min(max(req.top_k, 1), FETCH_CAP)
    logger.info(
        "[TIMER] request_received %.3f ms strategy=%s query_groups=%s top_k=%s",
        0.0,
        req.strategy_id,
        len(req.query_groups),
        top_k,
    )
    if req.strategy_id not in _strategies:
        logger.info(
            "[TIMER] total_request %.3f ms strategy=%s status=not_found",
            (time.monotonic() - t0) * 1000,
            req.strategy_id,
        )
        raise HTTPException(404, f"Strategy '{req.strategy_id}' not found.")

    strategy = _strategies[req.strategy_id]

    try:
        results = await strategy.search(
            [g.model_dump() for g in req.query_groups],
            limit=top_k,
        )
    except TimeoutError as exc:
        logger.info(
            "[TIMER] total_request %.3f ms strategy=%s status=timeout",
            (time.monotonic() - t0) * 1000,
            req.strategy_id,
        )
        raise HTTPException(408, str(exc))
    except Exception as exc:
        logger.info(
            "[TIMER] total_request %.3f ms strategy=%s status=error error=%s",
            (time.monotonic() - t0) * 1000,
            req.strategy_id,
            exc,
        )
        raise HTTPException(500, f"Strategy execution error: {exc}")

    total_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "[TIMER] total_request %.3f ms strategy=%s status=ok results=%s",
        total_ms,
        req.strategy_id,
        len(results),
    )
    return {
        "results":           results[:top_k],
        "strategy_id":       req.strategy_id,
        "total":             min(len(results), top_k),
        "execution_time_ms": int(total_ms),
    }


# ── Raw-data endpoint (called by local-client DataProvider in LOCAL mode) ─────

@app.post("/api/raw-data")
async def raw_data(body: dict):
    query_groups = body.get("query_groups", [])
    limit = min(int(body.get("limit", 1000)), 1000)
    data = await _data_provider.get_raw_data(query_groups, limit=limit)
    return data
