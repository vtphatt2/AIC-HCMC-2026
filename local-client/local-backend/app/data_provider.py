import os
import json
import logging
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
        for path in (REPO_ROOT / "AIC2026_sample", REPO_ROOT.parent / "AIC2026_sample")
        if path.is_dir()
    ),
    REPO_ROOT / "AIC2026_sample",
)
SAMPLE_ROOT = Path(os.getenv("AIC_SAMPLE_ROOT", DEFAULT_SAMPLE_ROOT))
logger = logging.getLogger(__name__)


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
        self._feature_matrix = None
        self._feature_frame_ids = []
        self._frames_by_id = {}
        self._text_encoder = None

        if self.mode == "LOCAL":
            if not REMOTE_SERVER_URL:
                raise RuntimeError(
                    "ENV_MODE=LOCAL requires REMOTE_SERVER_URL to be set in .env\n"
                    "Example: REMOTE_SERVER_URL=https://xxxx.ngrok.io"
                )
            print(f"DataProvider: LOCAL mode → {REMOTE_SERVER_URL}")

        elif self.mode == "MOCK":
            self._videos, self._frames, self._ocr, self._transcripts = self._load_mock()
            print(
                f"DataProvider: MOCK mode — "
                f"{len(self._videos)} videos, {len(self._frames)} frames loaded"
            )

        elif self.mode == "SAMPLE":
            self._videos, self._frames = self._load_sample()
            self._frames_by_id = {frame["frame_id"]: frame for frame in self._frames}
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

    async def get_raw_data(self, query_groups: list[dict], limit: int = 1000) -> dict:
        """
        Returns a unified dict consumed by strategy.fusion_and_temporal():
          frames, ocr, transcripts (lists), videos (dict keyed by video_id)
        """
        if self.mode == "MOCK":
            return self._local_raw_data(limit)
        if self.mode == "SAMPLE":
            return self._sample_raw_data(query_groups, limit)
        return await self._fetch_from_remote(query_groups, limit)

    async def search_transcripts(self, query: str, limit: int = 10) -> list[dict]:
        """Search transcripts independently and return matched intervals with nearest frame info."""
        if self.mode == "MOCK":
            return self._transcript_search_mock(query, limit)
        if self.mode == "SAMPLE":
            return self._transcript_search_sample(query, limit)
        return await self._transcript_search_remote(query, limit)

    # ── MOCK ──────────────────────────────────────────────────────────────────

    def _load_mock(self):
        def load(name):
            return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))

        videos_list = load("mock_videos.json")
        frames = load("mock_frames.json")
        ocr = load("mock_ocr.json")
        transcripts = load("mock_transcripts.json")
        return videos_list, frames, ocr, transcripts

    def _local_raw_data(self, limit: int) -> dict:
        videos_by_id = {v["video_id"]: v for v in self._videos}
        return {
            "frames":      self._frames[:limit],
            "ocr":         self._ocr[:limit],
            "transcripts": self._transcripts[:limit],
            "videos":      videos_by_id,
        }

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

        for metadata_path in sorted(metadata_dir.glob("*.json")):
            video_id = metadata_path.stem
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            fps = float(data.get("fps") or 25.0)
            frame_paths = sorted((features_dir / video_id).glob("*.npy"))
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

    def _sample_raw_data(self, query_groups: list[dict], limit: int) -> dict:
        frames = []
        for group_index, group in enumerate(query_groups):
            semantic_query = str(group.get("semantic_query", "")).strip()
            if not semantic_query:
                continue

            query_vector = self._encode_sample_text(semantic_query)
            hits = self.linear_search_by_vector(query_vector, top_k=limit)
            for hit in hits:
                hit["_query_group_index"] = group_index
            frames.extend(hits)
            logger.info("SAMPLE semantic query group=%s returned %s hits", group_index, len(hits))

        if not frames:
            frames = self._frames[:limit]

        videos_by_id = {v["video_id"]: v for v in self._videos}
        return {
            "frames":      self._dedupe_sample_frames(frames),
            "ocr":         self._ocr[:limit],
            "transcripts": self._transcripts[:limit],
            "videos":      videos_by_id,
        }

    def _encode_sample_text(self, text: str):
        if self._text_encoder is None:
            from app.services.text_encoder import PECoreTextEncoder

            self._text_encoder = PECoreTextEncoder()
        return self._text_encoder.encode(text)

    def resolve_sample_frame_id(self, query: str) -> str | None:
        """Accept L01_V001_000022, L01_V001/000022, or L01_V001 000022."""
        if self.mode != "SAMPLE":
            return None

        query = query.strip().replace("\\", "/")
        if not query:
            return None

        candidates = [query]
        if "/" in query:
            video_id, frame = query.rsplit("/", 1)
            candidates.append(f"{video_id}_{Path(frame).stem}")
        parts = query.split()
        if len(parts) == 2:
            candidates.append(f"{parts[0]}_{Path(parts[1]).stem}")

        for candidate in candidates:
            if candidate in self._frames_by_id:
                return candidate
        return None

    def linear_search_by_frame_id(self, frame_id: str, top_k: int = 100) -> list[dict]:
        if self.mode != "SAMPLE":
            raise RuntimeError("linear_search_by_frame_id is only available in ENV_MODE=SAMPLE")
        self._ensure_feature_index()

        if frame_id not in self._feature_frame_ids:
            raise ValueError(f"Frame '{frame_id}' has no PECore feature vector")

        query_idx = self._feature_frame_ids.index(frame_id)
        query_vector = self._feature_matrix[query_idx]
        return self.linear_search_by_vector(query_vector, top_k=top_k)

    def linear_search_by_vector(self, query_vector, top_k: int = 100) -> list[dict]:
        if self.mode != "SAMPLE":
            raise RuntimeError("linear_search_by_vector is only available in ENV_MODE=SAMPLE")
        self._ensure_feature_index()

        import numpy as np

        query_vector = np.asarray(query_vector, dtype="float32").reshape(-1)
        if query_vector.shape[0] != self._feature_matrix.shape[1]:
            raise ValueError(
                f"Query vector has dim {query_vector.shape[0]}; "
                f"expected {self._feature_matrix.shape[1]}"
            )
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Query vector has zero norm")

        query_vector = query_vector / norm
        scores = self._feature_matrix @ query_vector
        top_k = max(1, min(int(top_k), len(scores)))
        partition_k = min(top_k - 1, len(scores) - 1)
        top_indices = np.argpartition(-scores, partition_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            hit_frame_id = self._feature_frame_ids[int(idx)]
            frame = dict(self._frames_by_id[hit_frame_id])
            frame["score"] = float(scores[int(idx)])
            results.append(frame)
        return results

    def _ensure_feature_index(self) -> None:
        if self._feature_matrix is not None:
            return

        import numpy as np
        from tqdm import tqdm

        vectors = []
        frame_ids = []
        for frame in tqdm(self._frames, desc="Building SAMPLE vector index", unit="frame"):
            feature_path = frame.get("_feature_path")
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

        self._feature_matrix = np.vstack(vectors)
        self._feature_frame_ids = frame_ids
        print(
            f"DataProvider: built linear PECore index — "
            f"{self._feature_matrix.shape[0]} vectors x {self._feature_matrix.shape[1]} dims"
        )

    @staticmethod
    def _dedupe_sample_frames(frames: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for frame in frames:
            identity = (frame.get("frame_id"), frame.get("_query_group_index"))
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(frame)
        return deduped

    # ── Transcript search ─────────────────────────────────────────────────────

    # rapidfuzz is optional — imported lazily on first use.
    _rapidfuzz_fuzz = None

    @classmethod
    def _ensure_rapidfuzz(cls) -> bool:
        """Try to import rapidfuzz.fuzz. Returns True if available."""
        if cls._rapidfuzz_fuzz is not None:
            return cls._rapidfuzz_fuzz is not False
        try:
            from rapidfuzz import fuzz as _fuzz
            cls._rapidfuzz_fuzz = _fuzz
            return True
        except ImportError:
            cls._rapidfuzz_fuzz = False
            return False

    @staticmethod
    def _score_transcript(text: str, query: str) -> tuple[float, str, str]:
        """
        Score a transcript segment against a normalized query.
        Returns (score, match_type, normalized_query).

        Priority: phrase exact match → token overlap → fuzzy
        match_type values: "phrase", "token_overlap", "fuzzy", "no_match"
        """
        from app.services.vietnamese_utils import normalize_vi, extract_keywords

        text_norm = normalize_vi(text)
        query_norm = extract_keywords(query)
        if not query_norm or not text_norm:
            return 0.0, "no_match", query_norm

        query_tokens = query_norm.split()
        num_tokens = len(query_tokens)

        # 1) Phrase match: full normalized query appears with word boundaries
        import re
        if re.search(r"\b" + re.escape(query_norm) + r"\b", text_norm):
            return 1.0, "phrase", query_norm

        # 2) Token overlap on normalized text.
        #    Score is capped at 0.8 so phrase matches always rank above.
        text_tokens = set(text_norm.split())
        matched = sum(1 for t in query_tokens if t in text_tokens)
        overlap_raw = round(matched / num_tokens, 4) if num_tokens else 0.0
        # Cap at 0.8 — phrase match (1.0) is the gold standard
        overlap_score = min(overlap_raw, 0.8)

        # Single-token queries: need at least 1 match
        if num_tokens == 1:
            if matched >= 1:
                return overlap_score, "token_overlap", query_norm
        else:
            # Multi-token: require >= 50% tokens matched
            if overlap_raw >= 0.5:
                return overlap_score, "token_overlap", query_norm

        # 3) Fuzzy fallback via rapidfuzz (if installed)
        if DataProvider._ensure_rapidfuzz() and DataProvider._rapidfuzz_fuzz:
            try:
                fuzzy_score = DataProvider._rapidfuzz_fuzz.token_set_ratio(
                    query_norm, text_norm
                ) / 100.0
                # Cap fuzzy below phrase score too
                fuzzy_score = min(fuzzy_score, 0.85)
                if num_tokens == 1:
                    fuzzy_threshold = 0.7
                elif num_tokens <= 3:
                    fuzzy_threshold = 0.5
                else:
                    fuzzy_threshold = 0.4
                if fuzzy_score >= fuzzy_threshold:
                    return round(fuzzy_score, 4), "fuzzy", query_norm
            except Exception:
                pass

        return 0.0, "no_match", query_norm

    def _find_nearest_frame(self, video_id: str, target_ms: int) -> dict | None:
        """Find the frame closest to target_ms for the given video_id."""
        if self.mode == "MOCK":
            candidates = [f for f in self._frames if f["video_id"] == video_id]
        elif self.mode == "SAMPLE":
            candidates = [f for f in self._frames if f["video_id"] == video_id]
        else:
            return None

        if not candidates:
            return None
        return min(candidates, key=lambda f: abs(int(f["timestamp_ms"]) - target_ms))

    @staticmethod
    def _nearest_frame_or_none(nearest: dict | None) -> dict:
        """Convert nearest frame to optional fields, using None for missing."""
        if nearest is None:
            return {
                "nearest_frame_id": None,
                "nearest_timestamp_ms": None,
                "frame_image_url": None,
            }
        return {
            "nearest_frame_id": nearest.get("frame_id"),
            "nearest_timestamp_ms": nearest.get("timestamp_ms"),
            "frame_image_url": nearest.get("image_url"),
        }

    @staticmethod
    def _deduplicate_nearby(results: list[dict], window_ms: int = 5000) -> list[dict]:
        """Drop results for the same video within `window_ms` of a higher-scored result."""
        if not results:
            return results
        kept = []
        # Track per-video kept timestamps
        video_kept: dict[str, list[int]] = {}
        for item in results:
            vid = item["video_id"]
            ts = item["start_time_ms"]
            if vid not in video_kept:
                video_kept[vid] = [ts]
                kept.append(item)
                continue
            # Check if too close to any kept timestamp for this video
            if any(abs(ts - kts) < window_ms for kts in video_kept[vid]):
                continue
            video_kept[vid].append(ts)
            kept.append(item)
        return kept

    def _transcript_search_mock(self, query: str, limit: int) -> list[dict]:
        videos_by_id = {v["video_id"]: v for v in self._videos}
        scored = []
        for t in self._transcripts:
            score, match_type, norm_q = self._score_transcript(t.get("text", ""), query)
            if score <= 0:
                continue
            vid = videos_by_id.get(t["video_id"], {})
            mid_ms = (int(t["start_time_ms"]) + int(t["end_time_ms"])) // 2
            nearest = self._find_nearest_frame(t["video_id"], mid_ms)
            scored.append({
                "video_id":            t["video_id"],
                "youtube_id":          vid.get("youtube_id", ""),
                "start_time_ms":       int(t["start_time_ms"]),
                "end_time_ms":         int(t["end_time_ms"]),
                "text":                t.get("text", ""),
                "score":               score,
                "match_type":          match_type,
                "normalized_query":    norm_q,
                "window_text":         "",
                "window_start_time_ms": None,
                "window_end_time_ms":  None,
                **self._nearest_frame_or_none(nearest),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = self._deduplicate_nearby(scored)
        return scored[:limit]

    def _transcript_search_sample(self, query: str, limit: int) -> list[dict]:
        """
        SAMPLE mode transcript search. Falls back to mock transcripts if no
        sample transcript files exist in AIC2026_sample/.
        """
        sample_transcript_dir = sample_subdir("transcripts")
        has_sample_transcripts = sample_transcript_dir.is_dir() and any(
            sample_transcript_dir.glob("*.txt")
        )

        if has_sample_transcripts:
            results = self._search_sample_transcript_files(
                sample_transcript_dir, query, limit
            )
            if results:
                return results

        logger.warning(
            "SAMPLE mode: no sample transcript files found at %s. "
            "Falling back to mock transcript data.",
            sample_transcript_dir,
        )
        return self._transcript_search_from_mock_file(query, limit)

    def _transcript_search_from_mock_file(self, query: str, limit: int) -> list[dict]:
        """Search mock transcripts loaded directly from the JSON file."""
        import json
        mock_transcripts_path = MOCK_DIR / "mock_transcripts.json"
        if not mock_transcripts_path.exists():
            logger.warning("Mock transcripts file not found: %s", mock_transcripts_path)
            return []

        mock_transcripts = json.loads(mock_transcripts_path.read_text(encoding="utf-8"))
        mock_videos_path = MOCK_DIR / "mock_videos.json"
        mock_frames_path = MOCK_DIR / "mock_frames.json"
        mock_videos = {}
        if mock_videos_path.exists():
            videos_list = json.loads(mock_videos_path.read_text(encoding="utf-8"))
            mock_videos = {v["video_id"]: v for v in videos_list}
        mock_frames = []
        if mock_frames_path.exists():
            mock_frames = json.loads(mock_frames_path.read_text(encoding="utf-8"))

        return self._score_transcript_entries(
            mock_transcripts, mock_videos, mock_frames, query, limit
        )

    def _score_transcript_entries(
        self, transcripts: list[dict], videos: dict, frames: list[dict],
        query: str, limit: int
    ) -> list[dict]:
        """Score a list of transcript entries against a query, with nearest-frame lookup."""
        scored = []
        for t in transcripts:
            score, match_type, norm_q = self._score_transcript(t.get("text", ""), query)
            if score <= 0:
                continue
            vid = videos.get(t["video_id"], {})
            mid_ms = (int(t["start_time_ms"]) + int(t["end_time_ms"])) // 2
            nearest = self._find_nearest_frame_from_list(frames, t["video_id"], mid_ms)
            scored.append({
                "video_id":            t["video_id"],
                "youtube_id":          vid.get("youtube_id", ""),
                "start_time_ms":       int(t["start_time_ms"]),
                "end_time_ms":         int(t["end_time_ms"]),
                "text":                t.get("text", ""),
                "score":               score,
                "match_type":          match_type,
                "normalized_query":    norm_q,
                "window_text":         "",
                "window_start_time_ms": None,
                "window_end_time_ms":  None,
                **self._nearest_frame_or_none(nearest),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = self._deduplicate_nearby(scored)
        return scored[:limit]

    @staticmethod
    def _find_nearest_frame_from_list(frames: list[dict], video_id: str, target_ms: int) -> dict | None:
        candidates = [f for f in frames if f["video_id"] == video_id]
        if not candidates:
            return None
        return min(candidates, key=lambda f: abs(int(f["timestamp_ms"]) - target_ms))

    def _search_sample_transcript_files(
        self, transcript_dir: Path, query: str, limit: int
    ) -> list[dict]:
        """Parse .txt transcript files from a directory, score with window context, deduplicate."""
        import re
        from app.services.vietnamese_utils import normalize_vi

        timestamp_re = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s+(.*)")

        videos_by_id = {v["video_id"]: v for v in self._videos}
        scored = []

        for filepath in sorted(transcript_dir.glob("*.txt")):
            video_id = filepath.stem
            if video_id.endswith("_Transcript"):
                video_id = video_id[: -len("_Transcript")]

            if video_id not in videos_by_id:
                continue

            # Parse all segments for this file into a list
            raw = filepath.read_text(encoding="utf-8")
            segments: list[dict] = []  # [{start_ms, end_ms, text}]
            for line in raw.splitlines():
                m = timestamp_re.match(line.strip())
                if not m:
                    continue
                h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
                txt = m.group(4).strip()
                s_ms = (h * 3600 + mm * 60 + ss) * 1000
                segments.append({
                    "start_ms": s_ms,
                    "end_ms": s_ms + 5000,  # will be refined below
                    "text": txt,
                })

            if not segments:
                continue

            # Set end_ms from the next segment's start_ms
            for idx in range(len(segments) - 1):
                segments[idx]["end_ms"] = segments[idx + 1]["start_ms"]
            # Last segment: keep 5-second window

            # Score each segment with window context
            for idx, seg in enumerate(segments):
                # Build window: segments[idx-1] + seg + segments[idx+1]
                window_parts = []
                win_start = seg["start_ms"]
                win_end = seg["end_ms"]

                if idx > 0:
                    prev = segments[idx - 1]
                    window_parts.append(prev["text"])
                    win_start = prev["start_ms"]
                window_parts.append(seg["text"])
                if idx < len(segments) - 1:
                    nxt = segments[idx + 1]
                    window_parts.append(nxt["text"])
                    win_end = nxt["start_ms"]

                window_text = " ".join(window_parts)

                # 1. Score main segment (phrase → token_overlap → fuzzy)
                seg_score, seg_match_type, norm_q = self._score_transcript(seg["text"], query)

                # 2. Check window for phrase match only (catches split sentences)
                win_phrase = False
                if seg_match_type != "phrase" and window_text:
                    from app.services.vietnamese_utils import normalize_vi, extract_keywords
                    w_norm = normalize_vi(window_text)
                    q_norm = extract_keywords(query)
                    import re
                    if q_norm and re.search(r"\b" + re.escape(q_norm) + r"\b", w_norm):
                        win_phrase = True

                # Use window phrase if main segment didn't phrase-match
                if win_phrase:
                    score = 1.0
                    match_type = "phrase"
                    # Keep window fields
                else:
                    score = seg_score
                    match_type = seg_match_type
                    window_text = ""
                    win_start = None
                    win_end = None

                if score <= 0:
                    continue

                vid = videos_by_id.get(video_id, {})
                nearest_frame = self._find_nearest_frame(video_id, seg["start_ms"])

                scored.append({
                    "video_id":            video_id,
                    "youtube_id":          vid.get("youtube_id", ""),
                    "start_time_ms":       seg["start_ms"],
                    "end_time_ms":         seg["end_ms"],
                    "text":                seg["text"],
                    "score":               score,
                    "match_type":          match_type,
                    "normalized_query":    norm_q,
                    "window_text":         window_text,
                    "window_start_time_ms": win_start,
                    "window_end_time_ms":  win_end,
                    **self._nearest_frame_or_none(nearest_frame),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = self._deduplicate_nearby(scored)
        return scored[:limit]

    async def _transcript_search_remote(self, query: str, limit: int) -> list[dict]:
        """Proxy transcript search to the remote server."""
        if not REMOTE_SERVER_URL:
            logger.warning("LOCAL mode but REMOTE_SERVER_URL not set. Returning empty results.")
            return []

        try:
            async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=10.0) as client:
                response = await client.post(
                    "/api/search-transcript",
                    json={"query": query, "top_k": limit},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning(
                    "Remote server does not have /api/search-transcript endpoint. "
                    "Returning empty results."
                )
                return []
            logger.error("Remote transcript search failed: %s", exc)
            return []
        except Exception as exc:
            logger.error("Remote transcript search error: %s", exc)
            return []

    @staticmethod
    def _youtube_id_from_link(link: str, fallback: str) -> str:
        parsed = urlparse(link)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/") or fallback
        return parse_qs(parsed.query).get("v", [fallback])[0] or fallback

    # ── LOCAL (proxy to remote server) ────────────────────────────────────────

    async def _fetch_from_remote(self, query_groups: list[dict], limit: int) -> dict:
        async with httpx.AsyncClient(base_url=REMOTE_SERVER_URL, timeout=15.0) as client:
            response = await client.post(
                "/api/raw-data",
                json={"query_groups": query_groups, "limit": limit},
            )
            response.raise_for_status()
            return response.json()
