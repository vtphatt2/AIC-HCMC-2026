import asyncio
import os
import json
import logging
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

ENV_MODE = os.getenv("ENV_MODE", "MOCK")
REMOTE_SERVER_URL = os.getenv("REMOTE_SERVER_URL", "").rstrip("/")

MOCK_DIR = Path(__file__).parent / "mock"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_ROOT = next(
    (
        path
        for path in (
            REPO_ROOT / "data",
            REPO_ROOT / "AIC2026_sample",
            REPO_ROOT.parent / "AIC2026_sample",
        )
        if path.is_dir()
    ),
    REPO_ROOT / "AIC2026_sample",
)
SAMPLE_ROOT = Path(os.getenv("AIC_SAMPLE_ROOT", DEFAULT_SAMPLE_ROOT))
logger = logging.getLogger(__name__)
CHANNELS = {
    "raw.semantic",
    "subtitled.semantic",
    "transcript.lexical",
    "transcript.semantic",
}
VISUAL_FEATURE_DIRS = {
    "raw.semantic": "raw_keyframe_embeddings",
    "subtitled.semantic": "subtitled_keyframe_embeddings",
}


def sample_subdir(name: str) -> Path:
    outer = SAMPLE_ROOT / name
    nested = outer / name
    if nested.is_dir():
        return nested
    return outer


class DataProvider:
    """
    Switches data source based on ENV_MODE:

      MOCK  — returns data from app/mock/*.json (no server needed)
      SAMPLE — reads AIC2026_sample keyframes/metadata directly on this machine
      LOCAL — proxies requests to the GPU server at REMOTE_SERVER_URL
    """

    def __init__(self):
        self.mode = ENV_MODE
        self._feature_indexes = {}
        self._feature_paths_by_channel = {}
        self._frames_by_id = {}
        self._videos_by_id = {}
        self._text_encoder = None
        self._transcript_chunks = None

        if self.mode == "LOCAL":
            if not REMOTE_SERVER_URL:
                raise RuntimeError(
                    "ENV_MODE=LOCAL requires REMOTE_SERVER_URL to be set in .env\n"
                    "Example: REMOTE_SERVER_URL=https://xxxx.ngrok.io"
                )
            print(f"DataProvider: LOCAL mode → {REMOTE_SERVER_URL}")

        elif self.mode == "MOCK":
            self._videos, self._frames, self._ocr, self._transcripts = self._load_mock()
            self._frames_by_id = {frame["frame_id"]: frame for frame in self._frames}
            self._videos_by_id = {video["video_id"]: video for video in self._videos}
            print(
                f"DataProvider: MOCK mode — "
                f"{len(self._videos)} videos, {len(self._frames)} frames loaded"
            )

        elif self.mode == "SAMPLE":
            self._videos, self._frames = self._load_sample()
            self._frames_by_id = {frame["frame_id"]: frame for frame in self._frames}
            self._videos_by_id = {video["video_id"]: video for video in self._videos}
            self._ocr = []
            self._transcripts = []
            print(
                f"DataProvider: SAMPLE mode — "
                f"{len(self._videos)} videos, {len(self._frames)} keyframes loaded"
            )

        else:
            raise RuntimeError(
                f"Unknown ENV_MODE='{self.mode}'. Valid values: MOCK, SAMPLE, LOCAL"
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
        if self.mode == "MOCK":
            return {"status": "skipped", "reason": "MOCK mode has no text encoder"}
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
    ) -> list[dict]:
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel '{channel}'. Available: {sorted(CHANNELS)}")
        query = str(query).strip()
        if not query:
            return []
        top_k = min(max(int(top_k), 1), 1000)

        if self.mode == "LOCAL":
            payload = {
                "channel": channel,
                "query": query,
                "top_k": top_k,
                "video_genre": video_genre,
            }
            if vector_search_algorithm:
                payload["vector_search_algorithm"] = vector_search_algorithm
            async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=15.0) as client:
                response = await client.post(
                    "/api/retrieve",
                    json=payload,
                )
                response.raise_for_status()
                return response.json().get("hits", [])

        if channel in VISUAL_FEATURE_DIRS:
            if self.mode == "MOCK":
                hits = [dict(frame) for frame in self._frames[:top_k]]
            else:
                hits = self.linear_search_by_vector(
                    self._encode_sample_text(query),
                    top_k=top_k,
                    channel=channel,
                )
            return self._rank_hits(channel, hits)
        if channel == "transcript.lexical":
            return self._search_local_transcripts(query, top_k)
        raise RuntimeError(
            "transcript.semantic is available through the remote SERVER; "
            f"it is not indexed in {self.mode} mode"
        )

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
        return [
            dict(frame)
            for frame in self._frames
            if frame.get("video_id") == video_id
            and int(start_ms) <= int(frame.get("timestamp_ms", -1)) <= int(end_ms)
        ][:limit]

    def results(self, hits: list[dict]) -> list[dict]:
        output = []
        seen = set()
        for hit in hits:
            frame_id = hit.get("frame_id")
            if not frame_id:
                raise ValueError("Final result hit must contain frame_id; call keyframes() for transcript chunks")
            if frame_id in seen:
                continue
            seen.add(frame_id)
            frame = {**self._frames_by_id.get(frame_id, {}), **hit}
            video = self._videos_by_id.get(frame.get("video_id"), {})
            confidence = float(frame.get("confidence", frame.get("score", 0.0)))
            row = {
                **frame,
                "video_id": frame.get("video_id", ""),
                "youtube_id": frame.get("youtube_id", video.get("youtube_id", "")),
                "frame_id": frame_id,
                "frame_number": int(frame.get("frame_number", 0)),
                "timestamp_ms": int(frame.get("timestamp_ms", 0)),
                "confidence": max(0.0, min(1.0, confidence)),
                "frame_image_url": frame.get("frame_image_url", frame.get("image_url", "")),
                "fps": float(frame.get("fps", video.get("fps", 25.0))),
            }
            output.append(_strip_private(row))
        return output

    async def search_transcript_chunks(self, query: str, limit: int = 100, topic_filter: str | None = None) -> list[dict]:
        """Search topic-based transcript chunks via remote server (vector search)."""
        if self.mode in ("MOCK", "SAMPLE"):
            return []
        return await self._transcript_chunks_search_remote(query, limit, topic_filter)

    async def _transcript_chunks_search_remote(self, query: str, limit: int, topic_filter: str | None = None) -> list[dict]:
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
                data = response.json()
                return data.get("results", [])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("Remote server does not have /api/search/transcript endpoint.")
                return []
            logger.error("Remote transcript chunk search failed: %s", exc)
            return []
        except Exception as exc:
            logger.error("Remote transcript chunk search error: %s", exc)
            return []

    # ── MOCK ──────────────────────────────────────────────────────────────────

    def _load_mock(self):
        def load(name):
            return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))

        videos_list = load("mock_videos.json")
        frames = load("mock_frames.json")
        ocr = load("mock_ocr.json")
        transcripts = load("mock_transcripts.json")
        return videos_list, frames, ocr, transcripts

    # ── SAMPLE ───────────────────────────────────────────────────────────────

    def _load_sample(self):
        metadata_dir = sample_subdir("metadata")
        keyframes_dir = sample_subdir("keyframes")
        features_dir = sample_subdir("PECore-features")

        for label, path in {
            "metadata": metadata_dir,
            "keyframes": keyframes_dir,
            "PECore-features": features_dir,
        }.items():
            if not path.is_dir():
                raise RuntimeError(f"AIC sample {label} directory not found: {path}")

        videos = []
        frames = []
        self._feature_paths_by_channel = {
            channel: {
                f"{video_dir.name}_{path.stem}": path
                for video_dir in sorted((features_dir / dirname).glob("*"))
                if video_dir.is_dir()
                for path in video_dir.glob("*.npy")
            }
            for channel, dirname in VISUAL_FEATURE_DIRS.items()
        }

        for metadata_path in sorted(metadata_dir.glob("*.json")):
            video_id = metadata_path.stem
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            fps = float(data.get("fps") or 25.0)
            frame_paths = sorted(
                (features_dir / VISUAL_FEATURE_DIRS["raw.semantic"] / video_id).glob("*.npy")
            )
            if not frame_paths:
                frame_paths = sorted((keyframes_dir / video_id).glob("*.jpg"))

            youtube_id = data.get("youtube_id") or self._youtube_id_from_link(
                data.get("video_link", ""),
                fallback=video_id,
            )
            max_timestamp_ms = 0

            for path in frame_paths:
                frame_number = int(path.stem)
                timestamp_ms = int(frame_number / fps * 1000)
                max_timestamp_ms = max(max_timestamp_ms, timestamp_ms)
                frames.append({
                    "frame_id":     f"{video_id}_{path.stem}",
                    "video_id":     video_id,
                    "frame_number": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "image_url":    f"/static/frames/{video_id}/{path.stem}.jpg",
                    "_feature_path": str(path) if path.suffix == ".npy" else "",
                })

            videos.append({
                "video_id":    video_id,
                "title":       data.get("title") or video_id,
                "youtube_id":  youtube_id,
                "fps":         fps,
                "duration_ms": max_timestamp_ms,
                "frame_count": len(frame_paths),
            })

        frames.sort(key=lambda f: (f["video_id"], f["frame_number"]))
        return videos, frames

    # ── SAMPLE linear vector search ──────────────────────────────────────────

    def _encode_sample_text(self, text: str):
        if self._text_encoder is None:
            from app.services.text_encoder import PECoreTextEncoder

            self._text_encoder = PECoreTextEncoder()
        return self._text_encoder.encode(text)

    def get_frame_and_video(self, frame_id: str) -> tuple[dict, dict] | None:
        """Look up frame/video metadata used by SAMPLE-mode image routes."""
        if self.mode != "SAMPLE":
            return None
        frame = self._frames_by_id.get(frame_id)
        if frame is None:
            return None
        video = self._videos_by_id.get(frame["video_id"])
        if video is None:
            return None
        return frame, video

    def linear_search_by_vector(
        self,
        query_vector,
        top_k: int = 100,
        channel: str = "raw.semantic",
    ) -> list[dict]:
        if self.mode != "SAMPLE":
            raise RuntimeError("linear_search_by_vector is only available in ENV_MODE=SAMPLE")
        self._ensure_feature_index(channel)
        matrix, frame_ids = self._feature_indexes[channel]

        import numpy as np

        query_vector = np.asarray(query_vector, dtype="float32").reshape(-1)
        if query_vector.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Query vector has dim {query_vector.shape[0]}; "
                f"expected {matrix.shape[1]}"
            )
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Query vector has zero norm")

        query_vector = query_vector / norm
        scores = matrix @ query_vector
        top_k = max(1, min(int(top_k), len(scores)))
        partition_k = min(top_k - 1, len(scores) - 1)
        top_indices = np.argpartition(-scores, partition_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            hit_frame_id = frame_ids[int(idx)]
            frame = dict(self._frames_by_id[hit_frame_id])
            frame["score"] = float(scores[int(idx)])
            results.append(frame)
        return results

    def _ensure_feature_index(self, channel: str = "raw.semantic") -> None:
        if channel not in VISUAL_FEATURE_DIRS:
            raise ValueError(f"'{channel}' is not a visual channel")
        if channel in self._feature_indexes:
            return

        import numpy as np
        from tqdm import tqdm

        vectors = []
        frame_ids = []
        paths = self._feature_paths_by_channel.get(channel, {})
        for frame in tqdm(self._frames, desc=f"Building {channel} index", unit="frame"):
            feature_path = paths.get(frame["frame_id"])
            if not feature_path:
                continue
            vector = np.load(feature_path).astype("float32").reshape(-1)
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            vectors.append(vector / norm)
            frame_ids.append(frame["frame_id"])

        if not vectors:
            raise RuntimeError("No PECore .npy feature vectors found for SAMPLE mode")

        matrix = np.vstack(vectors)
        self._feature_indexes[channel] = (matrix, frame_ids)
        print(
            f"DataProvider: built {channel} index — "
            f"{matrix.shape[0]} vectors x {matrix.shape[1]} dims"
        )

    @staticmethod
    def _rank_hits(channel: str, hits: list[dict]) -> list[dict]:
        return [
            {**hit, "channel": channel, "item_id": hit["frame_id"], "rank": rank}
            for rank, hit in enumerate(hits, start=1)
        ]

    def _search_local_transcripts(self, query: str, top_k: int) -> list[dict]:
        if self.mode == "MOCK":
            chunks = [
                {
                    "chunk_id": row.get("id", f"mock:{index}"),
                    "video_id": row["video_id"],
                    "start_time_ms": row["start_time_ms"],
                    "end_time_ms": row["end_time_ms"],
                    "text": row["text"],
                }
                for index, row in enumerate(self._transcripts)
            ]
        else:
            if self._transcript_chunks is None:
                from app.services.transcript_index import Transcript

                chunks = []
                for path in sorted(sample_subdir("transcripts").glob("*_Transcript.txt")):
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

    @staticmethod
    def _youtube_id_from_link(link: str, fallback: str) -> str:
        parsed = urlparse(link)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/") or fallback
        return parse_qs(parsed.query).get("v", [fallback])[0] or fallback

def _strip_private(value):
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value
