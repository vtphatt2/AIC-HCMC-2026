"""Where the local backend gets its data.

One dataset: the organizers' lot archives under `challenge_resources/data/zip_file/`.
Everything downstream of those archives is derived, never a second source of truth:

    zip_file/*_results.zip
      ├─ ingest_zip_pipeline_results.py  → Milvus (frame records)
      ├─ export_vectors_npy.py           → vectors.f32.npy   (what search reads)
      └─ export_video_fps.py             → video_fps.json    (frame ↔ timestamp)

    raw_zip/Videos_L*.zip                → the pictures and the video itself

The older `AIC2026_sample` layout (keyframes/, metadata/, PECore-features/ — nine
L01–L03 videos, no overlap with the ~193k indexed vectors) is gone, along with
the MOCK fixtures that shadowed it. It described videos nobody could search for,
which made "my query returns nothing" ambiguous in a way that cost more than the
fixtures were worth.
"""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)   # shared defaults (ports, ngrok, tuning)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)  # this service, wins

ENV_MODE = os.getenv("ENV_MODE", "ZIP").upper()
REMOTE_SERVER_URL = os.getenv("REMOTE_SERVER_URL", "").rstrip("/")
DATA_ROOT = Path(os.getenv("AIC_SAMPLE_ROOT", REPO_ROOT / "challenge_resources" / "data"))
logger = logging.getLogger(__name__)
CHANNELS = {
    "raw.semantic",
    "subtitled.semantic",
    "transcript.lexical",
    "transcript.semantic",
}
# Channels the lot archives carry no data for. They exist on remote-server, so a
# strategy written against all four still runs there — it just says so plainly
# here instead of returning an empty list that looks like "no matches".
REMOTE_ONLY_CHANNELS = {"subtitled.semantic", "transcript.semantic"}


def data_subdir(name: str) -> Path:
    outer = DATA_ROOT / name
    nested = outer / name
    return nested if nested.is_dir() else outer


class DataProvider:
    """
    ZIP   — searches the vectors exported from the lot archives, on this machine
    LOCAL — proxies every raw-data request to the GPU server at REMOTE_SERVER_URL
    """

    def __init__(self):
        self.mode = ENV_MODE
        self._text_encoder = None
        self._transcript_chunks = None
        self._milvus_collection = None
        self._numpy_store = False

        if self.mode == "LOCAL":
            if not REMOTE_SERVER_URL:
                raise RuntimeError(
                    "ENV_MODE=LOCAL requires REMOTE_SERVER_URL to be set in .env\n"
                    "Example: REMOTE_SERVER_URL=https://xxxx.ngrok.io"
                )
            print(f"DataProvider: LOCAL mode -> {REMOTE_SERVER_URL}")

        elif self.mode == "ZIP":
            # Preferred: exact, and ~45x faster than Milvus Lite's own brute
            # force over the same bytes (~35 ms vs ~1570 ms at top_k=1000).
            # Milvus Lite stays as the fallback so a machine that has not run
            # scripts/export_vectors_npy.py still works — but see
            # docs/archive/milvus-lite-hnsw-recall-bug.md for why that fallback must
            # not be pointed at an HNSW collection.
            from app.db import numpy_vector_store

            self._numpy_store = numpy_vector_store.available()
            if self._numpy_store:
                store = numpy_vector_store.get_store()
                print(
                    f"DataProvider: ZIP mode -> numpy memmap (exact), "
                    f"{store.vectors.shape[0]} vectors"
                )
                stale = numpy_vector_store.staleness_warning()
                if stale:
                    print(f"DataProvider: WARNING - {stale}")
            else:
                self._milvus_collection = self._connect_milvus_lite()
                if self._milvus_collection is not None:
                    print(
                        f"DataProvider: ZIP mode -> Milvus Lite, "
                        f"{self._milvus_collection.num_entities} vectors indexed"
                    )
                else:
                    raise RuntimeError(
                        "No searchable vectors found.\n"
                        "Run the ingest, then the export:\n"
                        "  cd remote-server && python scripts/ingest_zip_pipeline_results.py --skip-postgres\n"
                        "  cd ../local-client/local-backend && python scripts/export_vectors_npy.py\n"
                        "and set MILVUS_LITE_PATH in .env so the backend knows where they landed."
                    )

        else:
            raise RuntimeError(
                f"Unknown ENV_MODE='{self.mode}'. Valid values: ZIP, LOCAL"
            )

    # ── Public interface ───────────────────────────────────────────────────────

    async def warmup_text_encoder(self, query: str = "warmup query", passes: int = 1) -> dict:
        if self.mode == "LOCAL":
            async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=120.0) as client:
                response = await client.post(
                    "/api/warmup_text_encoder",
                    params={"passes": passes},
                )
                response.raise_for_status()
                return response.json()
        if self._text_encoder is None:
            from app.services.text_encoder import PECoreTextEncoder

            self._text_encoder = PECoreTextEncoder()
        return await asyncio.to_thread(self._text_encoder.warmup, query, passes)

    async def retrieve(
        self,
        channel: str,
        query: str,
        *,
        top_k: int = 100,
        video_genre: str = "All",
        vector_search_algorithm: str | None = None,
        exclude_frame_ids: list[str] | None = None,
    ) -> list[dict]:
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel '{channel}'. Available: {sorted(CHANNELS)}")
        query = str(query).strip()
        if not query:
            return []
        top_k = min(max(int(top_k), 1), 1000)
        excluded = list(dict.fromkeys(str(value) for value in (exclude_frame_ids or []) if value))

        if self.mode == "LOCAL":
            payload = {
                "channel": channel,
                "query": query,
                "top_k": top_k,
                "video_genre": video_genre,
            }
            if vector_search_algorithm:
                payload["vector_search_algorithm"] = vector_search_algorithm
            if excluded:
                payload["exclude_frame_ids"] = excluded
            async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=15.0) as client:
                response = await client.post("/api/retrieve", json=payload)
                response.raise_for_status()
                return response.json().get("hits", [])

        if channel in REMOTE_ONLY_CHANNELS:
            raise RuntimeError(
                f"'{channel}' is not in the lot archives — it is indexed on "
                f"remote-server. Run with ENV_MODE=LOCAL to reach it."
            )

        if channel == "transcript.lexical":
            return self._search_local_transcripts(query, top_k)

        t_encode = time.monotonic()
        query_vector = self._encode_text(query)
        logger.info(
            "[TIMER] text_encode %.3f ms text_len=%s",
            (time.monotonic() - t_encode) * 1000, len(query),
        )
        t_search = time.monotonic()
        if self._numpy_store:
            from app.db import numpy_vector_store

            hits = numpy_vector_store.vector_search(
                query_vector.tolist(),
                top_k=top_k,
                exclude_frame_ids=excluded,
                video_genre=video_genre,
                include_vector=True,
            )
            backend = "numpy"
        else:
            from app.db import milvus_client

            hits = milvus_client.vector_search(
                self._milvus_collection,
                query_vector.tolist(),
                top_k=top_k,
                algorithm=milvus_client.DEFAULT_ALGORITHM,
                expr=_exclude_frames_expr(excluded),
                include_vector=True,
            )
            backend = "milvus"
        logger.info(
            "[TIMER] vector_search %.3f ms backend=%s channel=%s top_k=%s excluded=%s hits=%s",
            (time.monotonic() - t_search) * 1000, backend, channel, top_k, len(excluded), len(hits),
        )
        # Derived here rather than trusted from whatever an ingest script stored:
        # this backend is the one that knows how *it* serves zip-sourced frames,
        # so a URL baked in at ingest time can only go stale.
        for hit in hits:
            hit["image_url"] = f"/api/zip-frame/{hit['video_id']}/{hit['timestamp_ms']}"
        return self._rank_hits(channel, hits)

    async def frame_embeddings(self, frame_ids: list[str]) -> dict[str, list[float]]:
        frame_ids = list(dict.fromkeys(str(frame_id) for frame_id in frame_ids if frame_id))
        if not frame_ids:
            return {}
        if self.mode == "LOCAL":
            async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=30.0) as client:
                response = await client.post(
                    "/api/frame-embeddings",
                    json={"frame_ids": frame_ids},
                )
                response.raise_for_status()
                return response.json().get("embeddings", {})
        if self._numpy_store:
            from app.db import numpy_vector_store

            return numpy_vector_store.frame_vectors(frame_ids)

        from app.db import milvus_client

        return milvus_client.query_frame_vectors(self._milvus_collection, frame_ids)

    async def keyframes(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
        *,
        limit: int = 20,
    ) -> list[dict]:
        if int(start_ms) > int(end_ms):
            raise ValueError("start_ms must be <= end_ms")
        limit = min(max(int(limit), 1), 1000)
        if self.mode == "LOCAL":
            async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=15.0) as client:
                response = await client.post(
                    "/api/keyframes",
                    json={
                        "video_id": video_id,
                        "start_ms": int(start_ms),
                        "end_ms": int(end_ms),
                        "limit": limit,
                    },
                )
                response.raise_for_status()
                return response.json().get("hits", [])
        if self._numpy_store:
            from app.db import numpy_vector_store

            hits = numpy_vector_store.frames_in_range(
                video_id, int(start_ms), int(end_ms), limit=limit
            )
        else:
            from app.db import milvus_client

            hits = milvus_client.query_frames_in_time_range(
                self._milvus_collection, video_id, int(start_ms), int(end_ms), limit=limit
            )
        for hit in hits:
            hit["image_url"] = f"/api/zip-frame/{hit['video_id']}/{hit['timestamp_ms']}"
        return hits

    def results(self, hits: list[dict]) -> list[dict]:
        from app.services.zip_frame_source import ingest_fps

        output = []
        seen = set()
        for hit in hits:
            frame_id = hit.get("frame_id")
            if not frame_id:
                raise ValueError(
                    "Final result hit must contain frame_id; call keyframes() for transcript chunks"
                )
            if frame_id in seen:
                continue
            seen.add(frame_id)
            video_id = hit.get("video_id", "")
            confidence = float(hit.get("confidence", hit.get("score", 0.0)))
            row = {
                **hit,
                "video_id": video_id,
                "youtube_id": hit.get("youtube_id", ""),
                "frame_id": frame_id,
                "frame_number": int(hit.get("frame_number", 0)),
                "timestamp_ms": int(hit.get("timestamp_ms", 0)),
                "confidence": max(0.0, min(1.0, confidence)),
                "frame_image_url": hit.get("frame_image_url", hit.get("image_url", "")),
                # The same fps the ingest used, so the frontend's frame counter
                # agrees with the frame_number the search returned.
                "fps": float(hit.get("fps") or ingest_fps(video_id) or 25.0),
            }
            output.append(_strip_private(row))
        return output

    async def search_transcript_chunks(
        self, query: str, limit: int = 100, topic_filter: str | None = None
    ) -> list[dict]:
        """Topic-based transcript chunk search — indexed on remote-server only."""
        if self.mode != "LOCAL":
            return []
        return await self._transcript_chunks_search_remote(query, limit, topic_filter)

    async def _transcript_chunks_search_remote(
        self, query: str, limit: int, topic_filter: str | None = None
    ) -> list[dict]:
        if not REMOTE_SERVER_URL:
            logger.warning("LOCAL mode but REMOTE_SERVER_URL not set. Returning empty results.")
            return []
        payload: dict = {"query": query, "top_k": limit}
        if topic_filter:
            payload["topic_filter"] = topic_filter
        try:
            async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=10.0) as client:
                response = await client.post("/api/search/transcript", json=payload)
                response.raise_for_status()
                return response.json().get("results", [])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("Remote server does not have /api/search/transcript endpoint.")
                return []
            logger.error("Remote transcript chunk search failed: %s", exc)
            return []
        except Exception as exc:
            logger.error("Remote transcript chunk search error: %s", exc)
            return []

    # ── Internals ─────────────────────────────────────────────────────────────

    def _connect_milvus_lite(self):
        """Fallback vector search for a machine that has not run
        scripts/export_vectors_npy.py — same app/db/milvus_client.py and schema
        as remote-server, reading the file the ingest wrote. Enabled by
        MILVUS_LITE_PATH."""
        if not os.getenv("MILVUS_LITE_PATH", "").strip():
            return None
        try:
            from app.db import milvus_client

            milvus_client.connect()
            # Follows VECTOR_SEARCH_BACKEND rather than hard-wiring "hnsw" —
            # milvus_lite 3.2.0's HNSW search returns wrong neighbours
            # (docs/archive/milvus-lite-hnsw-recall-bug.md), so this must land on flat.
            algorithm = milvus_client.DEFAULT_ALGORITHM
            if not milvus_client.has_collection_for_algorithm(algorithm, "raw.semantic"):
                print(
                    f"DataProvider: MILVUS_LITE_PATH is set, but no raw.semantic "
                    f"'{algorithm}' collection exists yet at that path "
                    f"(run an ingest script with --vector-index {algorithm})"
                )
                return None
            print(f"DataProvider: Milvus Lite vector search using '{algorithm}'")
            return milvus_client.get_collection_for_name(
                milvus_client.collection_name_for_algorithm(algorithm, "raw.semantic")
            )
        except Exception as exc:
            print(f"DataProvider: MILVUS_LITE_PATH is set but connection failed: {exc}")
            return None

    def _encode_text(self, text: str):
        if self._text_encoder is None:
            from app.services.text_encoder import PECoreTextEncoder

            self._text_encoder = PECoreTextEncoder()
        return self._text_encoder.encode(text)

    @staticmethod
    def _rank_hits(channel: str, hits: list[dict]) -> list[dict]:
        return [
            {**hit, "channel": channel, "item_id": hit["frame_id"], "rank": rank}
            for rank, hit in enumerate(hits, start=1)
        ]

    def _search_local_transcripts(self, query: str, top_k: int) -> list[dict]:
        if self._transcript_chunks is None:
            from app.services.transcript_index import Transcript

            chunks = []
            for path in sorted(data_subdir("transcripts").glob("*_Transcript.txt")):
                transcript = Transcript.from_txt(path)
                for index, segment in enumerate(transcript.segments):
                    chunks.append({
                        "chunk_id": f"{transcript.video_id}:{index}",
                        "video_id": transcript.video_id,
                        "start_time_ms": segment.start_ms,
                        "end_time_ms": segment.end_ms,
                        "text": segment.text,
                    })
            self._transcript_chunks = chunks
        chunks = self._transcript_chunks

        tokens = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for chunk in chunks:
            text_tokens = set(re.findall(r"\w+", chunk["text"].casefold()))
            score = len(tokens & text_tokens) / max(1, len(tokens))
            if score:
                scored.append(({**chunk, "score": score}, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            {**hit, "channel": "transcript.lexical", "rank": rank}
            for rank, (hit, _) in enumerate(scored[:top_k], start=1)
        ]


def _exclude_frames_expr(frame_ids: list[str]) -> str | None:
    if not frame_ids:
        return None
    quoted = ", ".join(json.dumps(frame_id) for frame_id in frame_ids)
    return f"frame_id not in [{quoted}]"


def _strip_private(value):
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value
