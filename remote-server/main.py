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
from app.services.translation import TranslationService

STRATEGIES_DIR = Path(__file__).parent / "app" / "strategies"
_strategies: dict[str, BaseStrategy] = {}
_data_provider: DataProvider | None = None
_translation_service = TranslationService()
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
    if os.getenv("WARMUP_TEXT_ENCODER", "false").lower() in {"1", "true", "yes"}:
        print("Warming up text encoder...")
        await _data_provider.warmup_text_encoder()
    if os.getenv("WARMUP_TRANSLATION", "false").lower() in {"1", "true", "yes"}:
        print("Warming up translation...")
        provider = os.getenv("TRANSLATION_PROVIDER", "nmt")
        try:
            await asyncio.to_thread(
                _translation_service.translate,
                ["khởi động"],
                provider,
            )
        except Exception:
            logger.exception("Translation warmup failed provider=%s", provider)

    print("Discovering strategies...")
    _strategies = discover_strategies(_data_provider)
    print(f"Server ready — {len(_strategies)} strategy/strategies loaded.")

    yield

    if _data_provider is not None:
        _data_provider.close()
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
    vector_search_algorithm: str | None = None


class TranslationRequest(BaseModel):
    texts: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "env_mode": os.getenv("ENV_MODE"),
        "vector_search_backend": os.getenv("VECTOR_SEARCH_BACKEND", "milvus"),
        "milvus_collections": {
            algorithm: milvus_client.collection_name_for_algorithm(algorithm)
            for algorithm in ("hnsw", "flat", "scann")
        },
        "pecore_device": os.getenv("PECORE_DEVICE", "cpu"),
        "pecore_precision": os.getenv("PECORE_PRECISION", "fp32"),
        "strategies": len(_strategies),
    }


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
async def warmup_text_encoder(passes: int = 10):
    if _data_provider is None:
        raise HTTPException(503, "DataProvider is not ready.")

    t0 = time.monotonic()
    try:
        result = await _data_provider.warmup_text_encoder(
            "warmup query",
            passes=min(max(passes, 1), 10),
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


@app.get("/api/vector-search-algorithms")
async def list_vector_search_algorithms():
    algorithms = _vector_search_algorithm_options()
    return {
        "default": _default_vector_algorithm(),
        "algorithms": algorithms,
    }


@app.post("/api/translate")
async def translate(req: TranslationRequest):
    provider = os.getenv("TRANSLATION_PROVIDER", "nmt")
    try:
        translations = await asyncio.to_thread(
            _translation_service.translate,
            req.texts,
            provider,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Translation failed provider=%s", provider)
        raise HTTPException(502, f"Translation failed: {exc}")

    return {"translations": translations}


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
    query_groups = [g.model_dump() for g in req.query_groups]
    if req.vector_search_algorithm:
        algorithm = _normalize_vector_algorithm(req.vector_search_algorithm)
        for group in query_groups:
            group["_vector_search_algorithm"] = algorithm

    try:
        results = await strategy.search(
            query_groups,
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
    if body.get("vector_search_algorithm"):
        algorithm = _normalize_vector_algorithm(str(body["vector_search_algorithm"]))
        for group in query_groups:
            group["_vector_search_algorithm"] = algorithm
    data = await _data_provider.get_raw_data(query_groups, limit=limit)
    return data


def _normalize_vector_algorithm(value: str) -> str:
    algorithm = value.strip().lower()
    if algorithm not in {"hnsw", "flat", "cagra", "scann"}:
        raise HTTPException(400, "vector_search_algorithm must be 'hnsw', 'flat', 'cagra', or 'scann'.")
    return algorithm


def _default_vector_algorithm() -> str:
    configured = os.getenv("VECTOR_SEARCH_BACKEND", "milvus").lower()
    if configured == "flat":
        return "flat"
    if configured == "cagra":
        return "cagra"
    if configured == "scann":
        return "scann"
    return "hnsw"


def _vector_search_algorithm_options() -> list[dict]:
    available = milvus_client.available_milvus_algorithms()
    return [
        {
            "id": "hnsw",
            "name": "HNSW",
            "available": available.get("hnsw", False),
            "collection": milvus_client.collection_name_for_algorithm("hnsw"),
            "description": "Milvus HNSW ANN search from the vector collection.",
        },
        {
            "id": "scann",
            "name": "ScaNN",
            "available": available.get("scann", False),
            "collection": milvus_client.collection_name_for_algorithm("scann"),
            "description": "Milvus ScaNN approximate search with raw vector reordering.",
        },
        {
            "id": "flat",
            "name": "FLAT",
            "available": available.get("flat", False),
            "collection": milvus_client.collection_name_for_algorithm("flat"),
            "description": "Milvus FLAT exact CPU search from the vector collection.",
        },
        {
            "id": "cagra",
            "name": "CAGRA",
            "available": _cagra_available(),
            "description": "NVIDIA GPU ANN search from the prepared cuVS CAGRA index.",
        },
    ]


def _cagra_available() -> bool:
    if importlib.util.find_spec("cuvs") is None or importlib.util.find_spec("cupy") is None:
        return False
    index_path = Path(os.getenv("CAGRA_INDEX_PATH", "cache/cagra/index.bin"))
    frame_ids_path = Path(os.getenv("CAGRA_FRAME_IDS_PATH", "cache/cagra/frame_ids.txt"))
    if not index_path.is_absolute():
        index_path = Path(__file__).parent / index_path
    if not frame_ids_path.is_absolute():
        frame_ids_path = Path(__file__).parent / frame_ids_path
    return index_path.is_file() and frame_ids_path.is_file()
