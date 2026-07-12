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

FRAME_IMAGE_SOURCE = os.getenv("FRAME_IMAGE_SOURCE", "local").strip().lower()

if FRAME_IMAGE_SOURCE == "youtube_storyboard":
    # WORKAROUND — see docs/youtube-storyboard-thumbnails-workaround.md.
    # Approximates the frame at each timestamp using YouTube's own scrubber
    # storyboard sprites, for when real dataset keyframes aren't present
    # locally. Not pixel-accurate; do not rely on this in production.
    from app.services.youtube_thumbnail import StoryboardUnavailable, get_thumbnail_jpeg

    @app.get("/static/frames/{video_id}/{frame_file}")
    async def storyboard_frame(video_id: str, frame_file: str):
        if _data_provider is None:
            raise HTTPException(503, "Backend not ready")

        frame_stem = Path(frame_file).stem
        lookup = _data_provider.get_frame_and_video(f"{video_id}_{frame_stem}")
        if lookup is None:
            raise HTTPException(404, f"Unknown frame {video_id}/{frame_file}")
        frame, video = lookup
        youtube_id = video.get("youtube_id")
        if not youtube_id:
            raise HTTPException(404, f"No youtube_id for video {video_id}")

        try:
            jpeg_bytes = get_thumbnail_jpeg(youtube_id, int(frame["timestamp_ms"]))
        except StoryboardUnavailable as exc:
            raise HTTPException(502, str(exc)) from exc

        return Response(content=jpeg_bytes, media_type="image/jpeg")

elif FRAME_IMAGE_SOURCE == "youtube_precise":
    # Sharper + timestamp-accurate than the storyboard crop above: resolves
    # the real video stream via yt-dlp and seeks into it with ffmpeg. Slower
    # per frame on a cache miss (network + seek + decode), so this runs in a
    # worker thread per request instead of blocking the event loop — a
    # results grid's images load concurrently rather than one at a time.
    # See docs/youtube-storyboard-thumbnails-workaround.md.
    import asyncio
    from app.services.youtube_precise_frame import PreciseFrameUnavailable, get_precise_frame_jpeg
    from app.services.youtube_thumbnail import StoryboardUnavailable, get_thumbnail_jpeg

    @app.get("/static/frames/{video_id}/{frame_file}")
    async def precise_frame(video_id: str, frame_file: str):
        if _data_provider is None:
            raise HTTPException(503, "Backend not ready")

        frame_stem = Path(frame_file).stem
        lookup = _data_provider.get_frame_and_video(f"{video_id}_{frame_stem}")
        if lookup is None:
            raise HTTPException(404, f"Unknown frame {video_id}/{frame_file}")
        frame, video = lookup
        youtube_id = video.get("youtube_id")
        if not youtube_id:
            raise HTTPException(404, f"No youtube_id for video {video_id}")

        try:
            jpeg_bytes = await asyncio.to_thread(
                get_precise_frame_jpeg, youtube_id, int(frame["timestamp_ms"])
            )
        except PreciseFrameUnavailable as exc:
            raise HTTPException(502, str(exc)) from exc

        return Response(content=jpeg_bytes, media_type="image/jpeg")

    # Fast, blurry placeholder counterpart to the route above — same
    # storyboard-crop workaround as FRAME_IMAGE_SOURCE=youtube_storyboard,
    # always available here so the frontend can show *something* instantly
    # while the precise frame extracts in the background, then swap to it.
    @app.get("/static/frames-preview/{video_id}/{frame_file}")
    async def precise_frame_preview(video_id: str, frame_file: str):
        if _data_provider is None:
            raise HTTPException(503, "Backend not ready")

        frame_stem = Path(frame_file).stem
        lookup = _data_provider.get_frame_and_video(f"{video_id}_{frame_stem}")
        if lookup is None:
            raise HTTPException(404, f"Unknown frame {video_id}/{frame_file}")
        frame, video = lookup
        youtube_id = video.get("youtube_id")
        if not youtube_id:
            raise HTTPException(404, f"No youtube_id for video {video_id}")

        try:
            jpeg_bytes = await asyncio.to_thread(
                get_thumbnail_jpeg, youtube_id, int(frame["timestamp_ms"])
            )
        except StoryboardUnavailable as exc:
            raise HTTPException(502, str(exc)) from exc

        return Response(content=jpeg_bytes, media_type="image/jpeg")

elif SAMPLE_KEYFRAMES_DIR.is_dir():
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


def _annotate_and_prefetch_precise_frames(results: list[dict]) -> None:
    """No-op unless FRAME_IMAGE_SOURCE=youtube_precise. Otherwise:
    (1) tags each result (and each temporal-cluster step) with a fast
        frame_preview_url the frontend can show immediately while the
        precise frame is still extracting, and
    (2) kicks off background resolution of every distinct video's stream
        URL right now, so the (slow, ~1-4s) yt-dlp lookup mostly finishes
        before the frontend ever requests a thumbnail image, instead of
        happening lazily on the first image request.
    Best-effort: failures here must never break the search response itself.
    """
    if FRAME_IMAGE_SOURCE != "youtube_precise":
        return

    import asyncio
    from app.services.youtube_precise_frame import prefetch_stream_url
    from app.services.youtube_thumbnail import prefetch_storyboard_index

    youtube_ids: set[str] = set()

    def tag(row: dict) -> None:
        image_url = row.get("frame_image_url") or ""
        if image_url.startswith("/static/frames/"):
            row["frame_preview_url"] = image_url.replace("/static/frames/", "/static/frames-preview/", 1)
        if row.get("youtube_id"):
            youtube_ids.add(row["youtube_id"])
        for step in row.get("steps") or []:
            tag(step)

    for row in results:
        tag(row)

    for youtube_id in youtube_ids:
        asyncio.create_task(asyncio.to_thread(prefetch_stream_url, youtube_id))
        asyncio.create_task(asyncio.to_thread(prefetch_storyboard_index, youtube_id))


# ── Request / Response models ─────────────────────────────────────────────────

class QueryGroup(BaseModel):
    semantic_query: str = ""
    text_query: str = ""
    temporal_offset_ms: int = 0   # ms after the previous group's result window


class SearchRequest(BaseModel):
    strategy_id: str
    query_groups: list[QueryGroup]
    top_k: int = 100
    video_genre: str = "All"
    vector_search_algorithm: str | None = None


class TranscriptChunkSearchRequest(BaseModel):
    query: str
    top_k: int = 100
    topic_filter: str = ""


class TranslationRequest(BaseModel):
    texts: list[str]


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


@app.get("/api/transcript/{video_id}")
async def get_transcript(video_id: str):
    """Full transcript for one video (SAMPLE mode only) — fetched once per
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


@app.get("/api/vector-search-algorithms")
async def list_vector_search_algorithms():
    if os.getenv("ENV_MODE", "MOCK").upper() == "LOCAL":
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
            "description": "Local SAMPLE exact search over .npy vectors.",
        }
    ]
    return {
        "default": "linear",
        "algorithms": algorithms,
    }


@app.post("/api/translate")
async def translate(req: TranslationRequest):
    """Proxy translation to the remote server when local-backend runs in LOCAL mode."""
    if os.getenv("ENV_MODE", "MOCK").upper() != "LOCAL":
        raise HTTPException(501, "Translation requires ENV_MODE=LOCAL or direct remote-server mode.")

    remote_base = os.getenv("REMOTE_SERVER_URL", "").rstrip("/")
    if not remote_base:
        raise HTTPException(500, "REMOTE_SERVER_URL is required for translation proxy.")

    async with httpx.AsyncClient(base_url=remote_base, timeout=30.0) as client:
        try:
            response = await client.post(
                "/api/translate",
                json=req.model_dump(),
                headers={"ngrok-skip-browser-warning": "1"},
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Remote translation request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise HTTPException(response.status_code, detail)

    return response.json()


@app.post("/api/search")
async def search(req: SearchRequest):
    """Run a search with the selected strategy and return ranked results."""
    if req.strategy_id not in _strategies:
        raise HTTPException(404, f"Strategy '{req.strategy_id}' not found. Available: {list(_strategies)}")

    strategy = _strategies[req.strategy_id]
    t0 = time.monotonic()
    top_k = min(max(req.top_k, 1), FETCH_CAP)
    query_groups = [g.model_dump() for g in req.query_groups]
    if req.vector_search_algorithm:
        algorithm = req.vector_search_algorithm.strip().lower()
        for group in query_groups:
            group["_vector_search_algorithm"] = algorithm

    try:
        results = await strategy.search(
            query_groups,
            limit=top_k,
            video_genre=req.video_genre,
        )
    except TimeoutError as exc:
        raise HTTPException(408, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Strategy error: {exc}")

    results = results[:top_k]
    _annotate_and_prefetch_precise_frames(results)

    return {
        "results":           results,
        "strategy_id":       req.strategy_id,
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
    _annotate_and_prefetch_precise_frames(results)

    return {
        "results":           results,
        "total":             min(len(results), top_k),
        "execution_time_ms": int((time.monotonic() - t0) * 1000),
    }
