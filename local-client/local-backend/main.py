"""
AIC 2026 — Local Backend (Development Playground)
Run with: uvicorn main:app --reload --port 8000
"""
import asyncio
import logging
import os
import time
import importlib
import inspect
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from app.data_provider import DataProvider
from app.services.translation import TranslationService
from app.services.strategy_config import StrategyConfigStore
from app.strategies.base_strategy import BaseStrategy, FETCH_CAP
from app.services.query_parser import QueryParser
from app.services.remote_zip_proxy import RemoteZipVideoProxy, ZipVideoUnavailable

logger = logging.getLogger(__name__)
STRATEGIES_DIR = Path(__file__).parent / "app" / "strategies"
_strategies: dict[str, BaseStrategy] = {}
_data_provider: DataProvider | None = None
_translation_service = TranslationService()
_strategy_configs = StrategyConfigStore(
    Path(__file__).resolve().parents[2] / "challenge_resources" / "data" / "strategy-configs"
)
# Streams video playback straight from the organizer's remote ZIPs (see
# scripts/build_zip_video_index.py). Empty index -> every lookup 404s and
# VideoModal falls back to YouTube, so this is safe to leave always-on.
_zip_video_proxy = RemoteZipVideoProxy()
_zip_upstream_client = httpx.AsyncClient(timeout=30.0)


def discover_strategies(data_provider: DataProvider, parser=None) -> dict[str, BaseStrategy]:
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
                    instance = cls(data_provider, parser)
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
    if os.getenv("WARMUP_TEXT_ENCODER", "false").lower() in {"1", "true", "yes"}:
        print("Warming up text encoder...")
        t0 = time.monotonic()
        await _data_provider.warmup_text_encoder()
        print(f"Text encoder warm ({(time.monotonic() - t0) * 1000:.0f} ms)")
    print("Discovering strategies…")
    parser = QueryParser() if os.getenv("GEMINI_API_KEY", "").strip() else None
    _strategies = discover_strategies(_data_provider, parser)
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
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frames come from the organizers' video ZIPs (/api/zip-frame below). In LOCAL
# mode the remote server serves its own, so pass those through unchanged.
if os.getenv("ENV_MODE", "ZIP").upper() == "LOCAL":
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


class TranscriptChunkSearchRequest(BaseModel):
    query: str
    top_k: int = 100
    topic_filter: str = ""


class TranslationRequest(BaseModel):
    texts: list[str]


class StrategyConfigUpdate(BaseModel):
    weights: dict[str, float | list[float]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":     "ok",
        "env_mode":   os.getenv("ENV_MODE", "ZIP"),
        "strategies": len(_strategies),
        "zip_video_index": len(_zip_video_proxy),
    }


@app.get("/api/warmup_text_encoder")
@app.post("/api/warmup_text_encoder")
async def warmup_text_encoder(passes: int = 1):
    if _data_provider is None:
        raise HTTPException(503, "DataProvider is not ready.")
    try:
        return await _data_provider.warmup_text_encoder(
            "warmup query",
            passes=min(max(passes, 1), 10),
        )
    except Exception as exc:
        raise HTTPException(500, f"Text encoder warmup error: {exc}") from exc


@app.get("/api/transcript/{video_id}")
async def get_transcript(video_id: str):
    """Full transcript for one video — fetched once per
    video by VideoModal; the frontend looks up the active segment locally
    against the already-polled playback time instead of round-tripping on
    every tick. Search-driven transcript results are a separate, not-yet-
    built feature (see app/services/transcript_index.py)."""
    from app.services.transcript_index import load_or_build_transcript

    transcript = load_or_build_transcript(video_id)
    if transcript is None:
        raise HTTPException(404, f"No transcript found for video_id={video_id}")

    return {
        "video_id": transcript.video_id,
        "segments": [s.to_dict() for s in transcript.segments],
    }


@app.get("/api/zip-video/{video_id}")
async def zip_video(video_id: str, request: Request):
    """Proxy video playback straight from the organizer's remote ZIP archive:
    translate the browser's Range request into the matching byte range
    inside the upstream ZIP and stream it through unmodified. No download,
    no decode — the <video> element does its own seeking against this URL
    exactly like it would against a plain MP4. 404 (unknown video_id, or
    upstream refusing Range) is the expected signal for VideoModal to fall
    back to YouTube."""
    try:
        headers, upstream = await _zip_video_proxy.open_range(
            _zip_upstream_client, video_id, request.headers.get("range")
        )
    except ZipVideoUnavailable as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error opening zip-video stream for %s", video_id)
        raise HTTPException(500, f"Unexpected error opening video stream: {exc}") from exc

    async def body():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body(), status_code=206, headers=headers)


# Backstop against unbounded fan-out, not the real throttle — ffmpeg decodes
# are capped separately in zip_frame_source (_decode_semaphore). Most of the
# time in this route is spent on Range requests, so keeping it near the decode
# cap made a cold grid serialise its per-video moov downloads behind ffmpeg.
_zip_frame_semaphore = asyncio.Semaphore(int(os.getenv("ZIP_FRAME_CONCURRENCY", "12")))
ZIP_FRAME_TIMEOUT_SEC = 45.0

# A page of results can name the same frame more than once (a video appears at
# several ranks, temporal steps land on the same keyframe), and when the grid
# renders they all fire at the same instant. Without this, each one starts its
# own decode of the identical picture.
#
# Deliberately not a "wait 1-2s, then process the batch" queue: that would add
# its full delay to every thumbnail, including the ones that had no duplicate
# to merge with. Sharing in-flight work costs nothing and catches the same
# duplicates; the locality between *different* frames of one video is handled a
# layer down by the region cache in zip_frame_source.
_zip_frame_inflight: dict[tuple[str, str, int], asyncio.Future] = {}


async def _single_flight(key: tuple[str, str, int], factory):
    """Run factory() once for concurrent callers sharing a key."""
    existing = _zip_frame_inflight.get(key)
    if existing is not None:
        return await asyncio.shield(existing)

    task = asyncio.ensure_future(factory())
    _zip_frame_inflight[key] = task
    # Cleared by the task itself, not by whoever happened to start it. The
    # starter can be cancelled (browser aborts thumbnails constantly) while the
    # decode runs on for the others still waiting on it — clearing in a finally
    # here would drop a live entry and let the next caller start a duplicate.
    task.add_done_callback(lambda _: _zip_frame_inflight.pop(key, None))
    return await asyncio.shield(task)


@app.get("/api/zip-frame/{video_id}/{timestamp_ms}")
async def zip_frame(video_id: str, timestamp_ms: int):
    """Decode one JPEG frame straight from the organizer's remote ZIP — for
    videos ingested without local keyframe JPGs (see
    remote-server/scripts/ingest_zip_pipeline_results.py). Builds a per-video
    sample-table index once (cached in memory), then computes the exact
    compressed byte range for this frame and fetches it with a single Range
    request — see app/services/zip_frame_source.py for why (pointing ffmpeg
    at a URL and letting it probe the container costs several round trips
    per frame and doesn't hold up under concurrent grid loading).

    A result grid can render up to 100 cards at once, each requesting its own
    thumbnail. The semaphore caps how many decode at once; the rest just
    queue for a slot instead of piling on and dragging the whole batch down
    together.

    This route is the boundary where every failure must become an actual
    HTTP response — get_frame_jpeg itself guarantees bytes-or-
    ZipFrameUnavailable (see app/services/range_http_client.py for the
    retry/backoff behind that), but this still bounds total wall time and
    catches anything unexpected so a client is never left waiting with
    nothing coming back."""
    from app.services.zip_frame_source import ZipFrameUnavailable, get_frame_jpeg

    async def decode():
        async with _zip_frame_semaphore:
            return await get_frame_jpeg(_zip_upstream_client, video_id, timestamp_ms)

    try:
        jpeg_bytes = await asyncio.wait_for(
            _single_flight(("jpeg", video_id, timestamp_ms), decode),
            timeout=ZIP_FRAME_TIMEOUT_SEC,
        )
    except ZipFrameUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, f"Frame request timed out for {video_id}") from exc
    except Exception as exc:
        logger.exception("Unexpected error decoding zip-frame %s/%s", video_id, timestamp_ms)
        raise HTTPException(500, f"Unexpected error decoding frame: {exc}") from exc

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/zip-frame-plan/{video_id}/{timestamp_ms}")
async def zip_frame_plan(video_id: str, timestamp_ms: int):
    """Client-side decode mode: tell the browser which bytes hold this frame
    instead of decoding it here.

    The route above spawns an ffmpeg process per thumbnail, on this machine,
    for every viewer. This one does the parsing (cached per video) and returns
    a byte range plus the sample layout inside it; the browser fetches that
    range from /api/zip-bytes and runs WebCodecs on its own CPU. Decode cost
    then scales with the number of viewers instead of piling onto the host.

    Cheap enough not to need the decode semaphore — after the first frame of a
    video the moov index is cached, so this is pure arithmetic."""
    from app.services.zip_frame_source import ZipFrameUnavailable, get_frame_plan

    try:
        return await asyncio.wait_for(
            _single_flight(
                ("plan", video_id, timestamp_ms),
                lambda: get_frame_plan(_zip_upstream_client, video_id, timestamp_ms),
            ),
            timeout=ZIP_FRAME_TIMEOUT_SEC,
        )
    except ZipFrameUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, f"Frame plan timed out for {video_id}") from exc
    except Exception as exc:
        logger.exception("Unexpected error planning zip-frame %s/%s", video_id, timestamp_ms)
        raise HTTPException(500, f"Unexpected error planning frame: {exc}") from exc


@app.get("/api/zip-bytes/{video_id}/{region_start}/{length}")
async def zip_bytes(video_id: str, region_start: int, length: int):
    """Byte passthrough for a range a plan already described. No decoding, no
    parsing — the expensive half of the old path is simply gone.

    Not a general proxy: fetch_region checks the range against this video's own
    extent inside the archive, so it can't be aimed anywhere else."""
    from app.services.zip_frame_source import ZipFrameUnavailable, fetch_region

    try:
        data = await asyncio.wait_for(
            fetch_region(_zip_upstream_client, video_id, region_start, length),
            timeout=ZIP_FRAME_TIMEOUT_SEC,
        )
    except ZipFrameUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, f"Byte range timed out for {video_id}") from exc
    except Exception as exc:
        logger.exception("Unexpected error fetching zip bytes %s", video_id)
        raise HTTPException(500, f"Unexpected error fetching bytes: {exc}") from exc

    # Immutable by construction — a byte range of an archive that never changes.
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


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
            "configurable": bool(s.config_schema),
        }
        for sid, s in _strategies.items()
    ]


def _strategy_or_404(strategy_id: str) -> BaseStrategy:
    strategy = _strategies.get(strategy_id)
    if strategy is None:
        raise HTTPException(404, f"Strategy '{strategy_id}' not found")
    return strategy


@app.get("/api/strategies/{strategy_id}/configs")
async def list_strategy_configs(strategy_id: str):
    strategy = _strategy_or_404(strategy_id)
    try:
        configs = _strategy_configs.list(
            strategy_id, strategy.version, strategy.config_schema
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"strategy_id": strategy_id, "schema": strategy.config_schema, "configs": configs}


@app.put("/api/strategies/{strategy_id}/configs/{config_id}")
async def save_strategy_config(strategy_id: str, config_id: str, req: StrategyConfigUpdate):
    strategy = _strategy_or_404(strategy_id)
    try:
        return _strategy_configs.save(
            strategy_id, strategy.version, strategy.config_schema, config_id, req.weights
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/strategies/{strategy_id}/configs/{config_id}")
async def delete_strategy_config(strategy_id: str, config_id: str):
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
    if os.getenv("ENV_MODE", "ZIP").upper() == "LOCAL":
        remote_base = os.getenv("REMOTE_SERVER_URL", "").rstrip("/")
        if remote_base:
            async with httpx.AsyncClient(base_url=remote_base, timeout=10.0) as client:
                try:
                    response = await client.get(
                        "/api/vector-search-algorithms",
                        headers={"ngrok-skip-browser-warning": "1"},
                    )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPError:
                    pass
    algorithms = [
        {
            "id": "linear",
            "name": "Linear",
            "available": True,
            "description": "Exact search over the vectors exported from the lot archives.",
        }
    ]
    return {
        "default": "linear",
        "algorithms": algorithms,
    }


@app.post("/api/translate")
async def translate(req: TranslationRequest):
    try:
        translations = await asyncio.to_thread(_translation_service.translate, req.texts)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Translation failed: {exc}")
    return {"translations": translations}


@app.post("/api/search")
async def search(req: SearchRequest):
    """Run a search with the selected strategy and return ranked results."""
    if req.strategy_id not in _strategies:
        raise HTTPException(404, f"Strategy '{req.strategy_id}' not found. Available: {list(_strategies)}")

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
    t0 = time.monotonic()
    top_k = min(max(req.top_k, 1), FETCH_CAP)
    query_groups = [g.model_dump() for g in req.query_groups]
    logger.info(
        "[TIMER] request_received strategy=%s top_k=%s duplicate_threshold=%s query_groups=%s",
        req.strategy_id, top_k, req.duplicate_threshold, len(query_groups),
    )
    try:
        results = await strategy.search(
            query_groups,
            limit=top_k,
            video_genre=req.video_genre,
            vector_search_algorithm=req.vector_search_algorithm,
            options=effective_config,
            config_id=config["id"],
            config_revision=config["revision"],
            duplicate_threshold=req.duplicate_threshold,
        )
    except TimeoutError as exc:
        logger.info("[TIMER] total_request %.3f ms status=timeout", (time.monotonic() - t0) * 1000)
        raise HTTPException(408, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Strategy error: {exc}")

    results = results[:top_k]
    logger.info(
        "[TIMER] total_request %.3f ms strategy=%s status=ok results=%s",
        (time.monotonic() - t0) * 1000, req.strategy_id, len(results),
    )

    return {
        "results":           results,
        "strategy_id":       req.strategy_id,
        "config_id":         config["id"],
        "config_revision":   config["revision"],
        "effective_config":  effective_config,
        "duplicate_threshold": req.duplicate_threshold,
        "total":             len(results),
        "execution_time_ms": int((time.monotonic() - t0) * 1000),
    }


@app.post("/api/search/transcript")
async def search_transcript_chunks(req: TranscriptChunkSearchRequest):
    """Search topic-based transcript chunks (vector search)."""
    if not _data_provider:
        raise HTTPException(503, "DataProvider is not ready.")
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty.")

    t0 = time.monotonic()
    top_k = min(max(req.top_k, 1), FETCH_CAP)

    try:
        results = await _data_provider.search_transcript_chunks(
            req.query.strip(), limit=top_k, topic_filter=req.topic_filter or None,
        )
    except Exception as exc:
        raise HTTPException(500, f"Transcript chunk search error: {exc}")

    results = results[:top_k]

    return {
        "results":           results,
        "total":             min(len(results), top_k),
        "execution_time_ms": int((time.monotonic() - t0) * 1000),
    }
