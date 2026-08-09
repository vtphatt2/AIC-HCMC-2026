import json
import os
import re
import time
import importlib
import inspect
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse

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
from app.services.strategy_config import StrategyConfigStore
from app.services.transcript_search import TranscriptSearchService
from app.services.transcript_jsonl_reader import transcript_response
from app.services.query_parser import QueryParser
from scripts.sample_paths import default_sample_root, sample_subdir

STRATEGIES_DIR = Path(__file__).parent / "app" / "strategies"
REMOTE_ROOT = Path(__file__).parent
_strategies: dict[str, BaseStrategy] = {}
_data_provider: DataProvider | None = None
_translation_service = TranslationService()
_transcript_search_service: TranscriptSearchService | None = None
_strategy_configs = StrategyConfigStore(REMOTE_ROOT.parent / "challenge_resources" / "data" / "strategy-configs")
logger = logging.getLogger(__name__)

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _youtube_id_from_link(link: str) -> str:
    if not link:
        return ""
    try:
        parsed = urlparse(link)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/")
        return parse_qs(parsed.query).get("v", [""])[0]
    except Exception:
        return ""


def _load_video_metadata_from_disk() -> list[dict]:
    sample_root = default_sample_root(REMOTE_ROOT.parent)
    metadata_dir = sample_subdir(sample_root, "metadata")
    if not metadata_dir.is_dir():
        return []
    videos = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            video_id = path.stem
            youtube_id = data.get("youtube_id") or _youtube_id_from_link(data.get("video_link", ""))
            if youtube_id and not YOUTUBE_ID_PATTERN.fullmatch(youtube_id):
                continue
            videos.append({
                "video_id": video_id,
                "title": data.get("title") or video_id,
                "youtube_id": youtube_id or "",
                "fps": float(data.get("fps") or 25.0),
                "duration_ms": 0,
                "frame_count": 0,
            })
        except Exception:
            pass
    return videos


async def _auto_populate_video_metadata() -> int:
    videos = _load_video_metadata_from_disk()
    if not videos:
        return 0
    return await postgres_client.upsert_video_metadata(videos)


def discover_strategies(data_provider: DataProvider, parser=None) -> dict[str, BaseStrategy]:
    found = {}
    for path in sorted(STRATEGIES_DIR.glob("*.py")):
        if path.stem.startswith("_") or path.stem == "base_strategy":
            continue
        module_name = f"app.strategies.{path.stem}"
        try:
            module = importlib.import_module(module_name)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseStrategy) and cls is not BaseStrategy:
                    instance = cls(data_provider, parser)
                    found[path.stem] = instance
                    print(f"  ✓ {path.stem}  ({cls.name}  by {cls.author})")
                    break
        except Exception as exc:
            print(f"  ✗ {path.stem}: {exc}")
    return found


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _strategies, _data_provider, _transcript_search_service
    print("Initialising DB schema...")
    await postgres_client.init_schema()
    milvus_client.connect()
    milvus_client.create_collection_if_missing()

    populated = await _auto_populate_video_metadata()
    if populated:
        print(f"Auto-populated {populated} video metadata entries (with youtube_id)")

    print("Loading DataProvider...")
    _data_provider = DataProvider()
    if os.getenv("WARMUP_TEXT_ENCODER", "false").lower() in {"1", "true", "yes"}:
        print("Warming up text encoder...")
        await _data_provider.warmup_text_encoder()
    if os.getenv("WARMUP_TRANSLATION", "false").lower() in {"1", "true", "yes"}:
        print("Warming up translation...")
        try:
            await asyncio.to_thread(
                _translation_service.translate,
                ["khởi động"],
            )
        except Exception:
            logger.exception("Translation warmup failed")

    if os.getenv("TRANSCRIPT_CHUNK_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes"}:
        try:
            milvus_client.create_transcript_collection_if_missing()
            _transcript_search_service = TranscriptSearchService(keyframe_dir=FRAME_STATIC_DIR)
            print("Transcript chunk search service ready")
            if os.getenv("WARMUP_TRANSCRIPT_SEARCH", "true").lower() in {"1", "true", "yes"}:
                print("Warming up transcript search service model...")
                await asyncio.to_thread(_transcript_search_service.warmup)
        except Exception as exc:
            print(f"Transcript chunk search service unavailable: {exc}")
            _transcript_search_service = None

    print("Discovering strategies...")
    parser = QueryParser() if os.getenv("GEMINI_API_KEY", "").strip() else None
    _strategies = discover_strategies(_data_provider, parser)
    print(f"Server ready — {len(_strategies)} strategy/strategies loaded.")

    yield

    if _data_provider is not None:
        _data_provider.close()
    await postgres_client.close_pool()


app = FastAPI(title="AIC 2026 Remote Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _frame_static_dir() -> Path:
    configured = os.getenv("FRAME_STATIC_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"FRAME_STATIC_DIR does not exist or is not a directory: {path}")
        return path

    sample_root = default_sample_root(REMOTE_ROOT.parent)
    sample_keyframes = sample_subdir(sample_root, "keyframes")
    if sample_keyframes.is_dir():
        return sample_keyframes

    return REMOTE_ROOT / "static" / "frames"


# Serve frame images without requiring a duplicate copy under remote-server/static.
STATIC_DIR = REMOTE_ROOT / "static"
FRAME_STATIC_DIR = _frame_static_dir()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
if not FRAME_STATIC_DIR.exists():
    FRAME_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/frames", StaticFiles(directory=str(FRAME_STATIC_DIR)), name="frames")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryGroup(BaseModel):
    query: str = ""
    temporal_offset_ms: int = 0   # ms after the previous group's result window


class SearchRequest(BaseModel):
    strategy_id: str
    config_id: str = "default"
    config_overrides: dict[str, float | list[float]] | None = None
    query_groups: list[QueryGroup]
    top_k: int = 100
    video_genre: str = "All"
    vector_search_algorithm: str | None = None


class RetrieveRequest(BaseModel):
    channel: str
    query: str
    top_k: int = 100
    video_genre: str = "All"
    vector_search_algorithm: str | None = None


class KeyframesRequest(BaseModel):
    video_id: str
    start_ms: int
    end_ms: int
    limit: int = 20


class TranslationRequest(BaseModel):
    texts: list[str]


class TranscriptSearchRequest(BaseModel):
    query: str
    top_k: int = 100
    topic_filter: str = ""


class StrategyConfigUpdate(BaseModel):
    weights: dict[str, float | list[float]]


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
    clean_path = path.removeprefix("/").removeprefix("static/")
    if clean_path.startswith("frames/"):
        target = (FRAME_STATIC_DIR / clean_path.removeprefix("frames/")).resolve()
        source_dir = FRAME_STATIC_DIR
    else:
        target = (STATIC_DIR / clean_path).resolve()
        source_dir = STATIC_DIR
    return {
        "static_dir": str(STATIC_DIR.resolve()),
        "frame_static_dir": str(FRAME_STATIC_DIR.resolve()),
        "path": path,
        "target": str(target),
        "source_dir": str(source_dir.resolve()),
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


@app.get("/api/transcript/{video_id}")
async def get_transcript(video_id: str):
    payload = transcript_response(video_id)
    if payload is None:
        raise HTTPException(404, f"No transcript found for video_id={video_id}")
    return payload


@app.get("/api/strategies")
async def list_strategies():
    return [
        {
            "id":          sid,
            "name":        s.name,
            "description": s.description,
            "author":      s.author,
            "version":     s.version,
            "configurable": bool(s.config_schema),
        }
        for sid, s in _strategies.items()
    ]


def _strategy_or_404(strategy_id: str) -> BaseStrategy:
    strategy = _strategies.get(strategy_id)
    if strategy is None:
        raise HTTPException(404, f"Strategy '{strategy_id}' not found")
    return strategy


def _require_strategy_config_write():
    if os.getenv("ALLOW_STRATEGY_CONFIG_WRITES", "false").lower() not in {"1", "true", "yes"}:
        raise HTTPException(403, "Strategy config writes are disabled on this server")


@app.get("/api/strategies/{strategy_id}/configs")
async def list_strategy_configs(strategy_id: str):
    strategy = _strategy_or_404(strategy_id)
    try:
        configs = _strategy_configs.list(strategy_id, strategy.version, strategy.config_schema)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"strategy_id": strategy_id, "schema": strategy.config_schema, "configs": configs}


@app.put("/api/strategies/{strategy_id}/configs/{config_id}")
async def save_strategy_config(strategy_id: str, config_id: str, req: StrategyConfigUpdate):
    _require_strategy_config_write()
    strategy = _strategy_or_404(strategy_id)
    try:
        return _strategy_configs.save(
            strategy_id, strategy.version, strategy.config_schema, config_id, req.weights
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/strategies/{strategy_id}/configs/{config_id}")
async def delete_strategy_config(strategy_id: str, config_id: str):
    _require_strategy_config_write()
    _strategy_or_404(strategy_id)
    try:
        _strategy_configs.delete(strategy_id, config_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Config '{config_id}' not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "deleted"}


@app.get("/api/vector-search-algorithms")
async def list_vector_search_algorithms():
    algorithms = _vector_search_algorithm_options()
    return {
        "default": _default_vector_algorithm(),
        "algorithms": algorithms,
    }


@app.post("/api/translate")
async def translate(req: TranslationRequest):
    try:
        translations = await asyncio.to_thread(
            _translation_service.translate,
            req.texts,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(502, f"Translation failed: {exc}")

    return {"translations": translations}


@app.post("/api/retrieve")
async def retrieve(req: RetrieveRequest):
    if _data_provider is None:
        raise HTTPException(503, "DataProvider is not ready")
    try:
        hits = await _data_provider.retrieve(
            req.channel,
            req.query,
            top_k=req.top_k,
            video_genre=req.video_genre,
            vector_search_algorithm=req.vector_search_algorithm,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"hits": hits}


@app.post("/api/keyframes")
async def keyframes(req: KeyframesRequest):
    if _data_provider is None:
        raise HTTPException(503, "DataProvider is not ready")
    try:
        hits = await _data_provider.keyframes(
            req.video_id,
            req.start_ms,
            req.end_ms,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"hits": hits}


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
        config = _strategy_configs.get(
            req.strategy_id, strategy.version, strategy.config_schema, req.config_id
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Config '{req.config_id}' not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        effective_config = _strategy_configs.resolve(
            strategy.config_schema, config["weights"], req.config_overrides
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    query_groups = [g.model_dump() for g in req.query_groups]
    try:
        results = await strategy.search(
            query_groups,
            limit=top_k,
            video_genre=req.video_genre,
            vector_search_algorithm=(
                _normalize_vector_algorithm(req.vector_search_algorithm)
                if req.vector_search_algorithm
                else None
            ),
            options=effective_config,
            config_id=config["id"],
            config_revision=config["revision"],
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
        "config_id":         config["id"],
        "config_revision":   config["revision"],
        "effective_config":  effective_config,
        "total":             min(len(results), top_k),
        "execution_time_ms": int(total_ms),
    }


@app.post("/api/search/transcript")
async def search_transcript(req: TranscriptSearchRequest):
    if _transcript_search_service is None:
        raise HTTPException(503, "Transcript chunk search is not available.")
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty.")

    t0 = time.monotonic()
    top_k = min(max(req.top_k, 1), FETCH_CAP)

    try:
        results = await _transcript_search_service.search(
            req.query.strip(),
            top_k=top_k,
            topic_filter=req.topic_filter.strip() or None,
        )
    except Exception as exc:
        logger.exception("Transcript chunk search error")
        raise HTTPException(500, f"Transcript search error: {exc}")

    total_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "[TIMER] transcript_search %.3f ms query=%s results=%s",
        total_ms,
        req.query[:80],
        len(results),
    )
    return {
        "results":           results[:top_k],
        "total":             min(len(results), top_k),
        "execution_time_ms": int(total_ms),
    }


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
