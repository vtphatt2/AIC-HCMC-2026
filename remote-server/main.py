import os
import time
import importlib
import inspect
import logging
import asyncio
from bisect import bisect_left, bisect_right
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env", override=False)   # shared defaults (ports, ngrok, tuning)
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)  # this service, wins

from app.data_provider import DataProvider
from app.strategies.base_strategy import BaseStrategy, FETCH_CAP
from app.db import postgres_client, milvus_client
from app.services.translation import TranslationService
from app.services.strategy_config import StrategyConfigStore
from app.services.transcript_search import TranscriptSearchService
from app.services.transcript_jsonl_reader import transcript_response
from app.services.query_parser import QueryParser
from app.services import local_zip_media

STRATEGIES_DIR = Path(__file__).parent / "app" / "strategies"
REMOTE_ROOT = Path(__file__).parent
_strategies: dict[str, BaseStrategy] = {}
_data_provider: DataProvider | None = None
_translation_service = TranslationService()
_transcript_search_service: TranscriptSearchService | None = None
_strategy_configs = StrategyConfigStore(REMOTE_ROOT.parent / "challenge_resources" / "data" / "strategy-configs")
logger = logging.getLogger(__name__)

ZIP_FRAME_TIMEOUT_SEC = float(os.getenv("ZIP_FRAME_TIMEOUT_SEC", "45"))


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
    """Only for lots that shipped keyframe JPGs. Lot archives carry none, so the
    default is an empty directory and thumbnails come from /api/zip-frame."""
    configured = os.getenv("FRAME_STATIC_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"FRAME_STATIC_DIR does not exist or is not a directory: {path}")
        return path
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
    duplicate_threshold: float = Field(default=0.98, ge=0.0, le=1.0)


class RetrieveRequest(BaseModel):
    channel: str
    query: str
    top_k: int = 100
    video_genre: str = "All"
    vector_search_algorithm: str | None = None
    exclude_frame_ids: list[str] | None = None


class FrameEmbeddingsRequest(BaseModel):
    frame_ids: list[str]


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
        # Scans raw_zip_videos/ on first call — this is the check for whether the
        # /api/zip-frame and /api/zip-video routes have anything to serve.
        "local_zip_videos": await local_zip_media.available_video_count(),
        "raw_zip_dir": str(local_zip_media.raw_zip_dir()),
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


@app.get("/api/video/{video_id}")
async def get_video_info(video_id: str):
    """Look up one video by its exact video_id — for jumping straight to it
    (e.g. found via transcript search) without a search box, and for the
    submission dashboard's fps/youtube_id lookup (lib/submission's
    getVideoInfo). Mirrors local-backend/main.py's endpoint of the same
    name and response shape; the source here is PostgreSQL's `videos` table
    (fps/youtube_id) plus Milvus (a seed frame), not numpy_vector_store.

    Was missing entirely on this backend until now — every call 404'd, and
    the frontend's getVideoInfo() silently caught that into a hardcoded
    {fps: 25, youtubeId: undefined} fallback, which is what sent videos
    through the zip-video stream instead of YouTube and showed a wrong fps
    even for a video with a real YouTube embed (e.g. L21_V023, actually 30
    fps, DB default 25.0)."""
    rows = await postgres_client.fetch_video_metadata([video_id])
    if not rows:
        raise HTTPException(404, f"No metadata found for video_id={video_id}")
    video = rows[0]

    frame_id, frame_number, timestamp_ms = None, 0, 0
    try:
        collection = milvus_client.get_collection()
        frames = milvus_client.query_frames_in_time_range(collection, video_id, 0, 10**12, limit=1)
        if frames:
            frame = frames[0]
            frame_id = frame["frame_id"]
            frame_number = frame["frame_number"]
            timestamp_ms = frame["timestamp_ms"]
    except Exception:
        logger.warning("get_video_info: Milvus frame lookup failed for %s", video_id, exc_info=True)

    return {
        "video_id": video_id,
        "youtube_id": video.get("youtube_id") or None,
        "frame_id": frame_id,
        "frame_number": frame_number,
        "timestamp_ms": timestamp_ms,
        "fps": float(video.get("fps") or 25.0),
    }


# A video's own indexed keyframe count tops out around 784 in this dataset
# (see keyframe_selection.md's per-scene sampling caps) — generous enough
# headroom to fetch a whole video's frames in one query and slice locally.
_CONTEXT_FRAMES_FETCH_LIMIT = 2000


@app.get("/api/video/{video_id}/context-frames")
async def get_context_frames(video_id: str, start_ms: int, end_ms: int, expand: int = 20):
    """Up to `expand` indexed keyframes immediately before start_ms and
    after end_ms — lets Video view's per-video strip fill itself out with
    real neighboring frames when a search only matched a tight handful, so
    a 3-frame result doesn't read as "that's all there is" when the video
    actually has far more indexed nearby. Mirrors local-backend's endpoint
    of the same name/response shape; source here is Milvus + PostgreSQL
    instead of numpy_vector_store."""
    rows = await postgres_client.fetch_video_metadata([video_id])
    fps = float(rows[0].get("fps") or 25.0) if rows else 25.0

    collection = milvus_client.get_collection()
    frames = milvus_client.query_frames_in_time_range(
        collection, video_id, 0, 10**12, limit=_CONTEXT_FRAMES_FETCH_LIMIT
    )
    # query_frames_in_time_range already sorts by timestamp_ms.
    timestamps = [f["timestamp_ms"] for f in frames]
    lo = bisect_left(timestamps, start_ms)
    hi = bisect_right(timestamps, end_ms)
    expand = max(0, int(expand))

    return {
        "fps": fps,
        "before": frames[max(0, lo - expand):lo],
        "after": frames[hi:hi + expand],
    }


@app.get("/api/zip-video/{video_id}")
async def zip_video(video_id: str, request: Request):
    """Stream playback from a local `Videos_L*.zip` in raw_zip_videos/, translating the
    browser's Range request into a seek inside the archive. Nothing is unpacked
    and nothing is decoded — the <video> element seeks against this URL exactly
    as it would against a plain MP4.

    404 (no archive holds this video_id) is the expected signal for VideoModal
    to fall back to YouTube."""
    try:
        headers, body = await local_zip_media.open_range(
            video_id, request.headers.get("range")
        )
    except local_zip_media.LocalZipUnavailable as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error opening zip-video stream for %s", video_id)
        raise HTTPException(500, f"Unexpected error opening video stream: {exc}") from exc

    return StreamingResponse(body, status_code=206, headers=headers)


@app.get("/api/zip-frame/{video_id}/{timestamp_ms}")
async def zip_frame(video_id: str, timestamp_ms: int):
    """Decode one JPEG straight out of a local archive, for lots ingested
    without keyframe JPGs (ingest_zip_pipeline_results.py). The per-video sample
    table is parsed once and cached; each frame is then one seek plus one
    ffmpeg decode.

    This route is the boundary where every failure has to become an HTTP
    response — get_frame_jpeg guarantees bytes or LocalZipUnavailable, and the
    timeout bounds total wall time so a client is never left waiting."""
    try:
        jpeg_bytes = await asyncio.wait_for(
            local_zip_media.get_frame_jpeg(video_id, timestamp_ms),
            timeout=ZIP_FRAME_TIMEOUT_SEC,
        )
    except local_zip_media.LocalZipUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, f"Frame request timed out for {video_id}") from exc
    except Exception as exc:
        logger.exception("Unexpected error decoding zip-frame %s/%s", video_id, timestamp_ms)
        raise HTTPException(500, f"Unexpected error decoding frame: {exc}") from exc

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        # A frame of an archive that never changes.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


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
            exclude_frame_ids=req.exclude_frame_ids,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"hits": hits}


@app.post("/api/frame-embeddings")
async def frame_embeddings(req: FrameEmbeddingsRequest):
    if _data_provider is None:
        raise HTTPException(503, "DataProvider is not ready")
    if len(req.frame_ids) > 20_000:
        raise HTTPException(400, "frame_ids is limited to 20000 items")
    return {"embeddings": await _data_provider.frame_embeddings(req.frame_ids)}


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
            duplicate_threshold=req.duplicate_threshold,
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
        "duplicate_threshold": req.duplicate_threshold,
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
