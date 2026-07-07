import asyncio
import logging
import time

from app.strategies.base_strategy import BaseStrategy, FETCH_CAP, MULTI_STEP_FETCH_MIN, EXECUTION_TIMEOUT_SEC

logger = logging.getLogger(__name__)

GENRE_ALPHA_MAP: dict[str, float] = {
    "Thời sự":    0.50,
    "Giáo dục":   0.35,
    "Ẩm thực":    0.25,
    "Pháp luật":  0.45,
    "Công nghệ":  0.35,
    "Du lịch":    0.20,
    "Thể thao":   0.30,
    "Giải trí":   0.35,
    "Văn hóa":    0.40,
    "Sức khỏe":   0.40,
    "Kinh tế":    0.40,
    "Đời sống":   0.35,
    "Môi trường": 0.30,
    "Giao thông": 0.30,
}
DEFAULT_ALPHA = 0.35


def _frame_image_url(video_id: str, frame_id: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    frame_part = frame_id.rsplit("_", 1)[-1]
    return f"/static/frames/{video_id}/{frame_part}.jpg"


def _tokens(text: str) -> set[str]:
    return {part.lower() for part in text.replace("_", " ").split() if len(part.strip()) >= 2}


def _text_score(text: str, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    haystack = text.lower()
    matched = sum(1 for t in query_tokens if t in haystack)
    return matched / len(query_tokens)


def _compute_dynamic_alpha(
    transcript_score: float,
    visual_score: float,
    has_semantic: bool,
    genre: str = "All",
) -> float:
    if not has_semantic:
        return GENRE_ALPHA_MAP.get(genre, DEFAULT_ALPHA)

    if transcript_score < 0.05:
        return 0.0
    if visual_score < 0.05:
        return 1.0

    ratio = transcript_score / (transcript_score + visual_score + 1e-6)
    base_alpha = GENRE_ALPHA_MAP.get(genre, DEFAULT_ALPHA)
    alpha = base_alpha + 0.2 * (ratio - 0.5)
    return round(max(0.0, min(1.0, alpha)), 3)


def _reciprocal_rank_fusion(
    rankings: list[list[dict]],
    key_fn,
    k: int = 60,
) -> dict:
    scores: dict = {}
    for ranked_list in rankings:
        for rank, item in enumerate(ranked_list, start=1):
            key = key_fn(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


class TwoStageFusionStrategy(BaseStrategy):
    """
    Two-Stage Late Fusion Retrieval Pipeline.

    Stage 1 (Coarse): Transcript vector search (E5, 384-dim) → candidate video_ids
    Stage 2 (Fine): PE-Core visual search (1280-dim) filtered to candidate videos via Milvus expr

    Score fusion uses a dynamic weighted combination:
        Score_final = α · Score_transcript + (1 - α) · Score_visual

    where α is dynamically adjusted based on signal quality and genre.
    Falls back to pure visual search if Stage 1 returns 0 results.
    """

    name = "Two-Stage Fusion v1"
    description = (
        "Two-stage late fusion: Stage 1 uses transcript vector search to identify "
        "candidate videos, Stage 2 runs PE-Core visual search filtered to those videos. "
        "Dynamic α weighting based on signal quality and genre."
    )
    author = "Team AIC 2026"
    version = "1.0"

    async def search(self, query_groups: list[dict], limit: int = 100, video_genre: str = "All") -> list[dict]:
        processed = self.pre_process(query_groups)
        fetch_limit = min(max(int(limit), 1), FETCH_CAP)
        if len(processed) > 1:
            fetch_limit = max(fetch_limit, MULTI_STEP_FETCH_MIN)

        raw_data = await self.data_provider.get_raw_data_two_stage(
            processed,
            limit=fetch_limit,
            video_genre=video_genre,
            stage1_top_k=20,
        )

        try:
            timer_start = time.monotonic()
            results = await asyncio.wait_for(
                asyncio.to_thread(self.fusion_and_temporal, raw_data, processed),
                timeout=EXECUTION_TIMEOUT_SEC,
            )
            logger.info(
                "[TIMER] fusion %.3f ms strategy=%s results=%s",
                (time.monotonic() - timer_start) * 1000,
                self.__class__.__name__,
                len(results),
            )
        except asyncio.TimeoutError:
            logger.info(
                "[TIMER] fusion %.3f ms strategy=%s timeout=true",
                (time.monotonic() - timer_start) * 1000,
                self.__class__.__name__,
            )
            raise TimeoutError(
                f"[{self.name}] fusion_and_temporal() exceeded {EXECUTION_TIMEOUT_SEC}s. "
                "Check for infinite loops or very expensive operations."
            )

        return self.post_filter(results)

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        ocr = raw_data.get("ocr", [])
        transcripts = raw_data.get("transcripts", [])
        transcript_chunks = raw_data.get("transcript_chunks", [])
        videos = raw_data.get("videos", {})
        video_genre = raw_data.get("video_genre", "All")
        stage1_chunks = raw_data.get("stage1_chunks", [])
        stage1_video_ids = raw_data.get("stage1_video_ids", set())
        stage2_frames = raw_data.get("stage2_frames", [])

        text_query = " ".join(g.get("text_query", "") for g in query_groups).strip()
        has_semantic = any(g.get("semantic_query", "").strip() for g in query_groups)
        has_text = bool(text_query)
        query_tokens = _tokens(text_query)

        has_stage1 = bool(stage1_chunks)
        has_stage2 = bool(stage2_frames)

        if not has_stage1 and not has_stage2:
            return self._fallback_results(frames, videos)

        if not has_stage1:
            logger.info("TwoStageFusion: Stage 1 empty, falling back to visual-only")
            return self._visual_only_results(stage2_frames or frames, videos, has_semantic)

        if not has_stage2 and has_semantic:
            logger.info("TwoStageFusion: Stage 2 empty, using transcript-only with frame mapping")
            return self._transcript_only_results(stage1_chunks, frames, videos)

        transcript_ranked = self._build_transcript_ranking(stage1_chunks, videos)
        visual_ranked = self._build_visual_ranking(stage2_frames, videos, has_semantic)

        avg_transcript_score = (
            sum(item["_score"] for item in transcript_ranked) / len(transcript_ranked)
            if transcript_ranked else 0.0
        )
        avg_visual_score = (
            sum(item["_score"] for item in visual_ranked) / len(visual_ranked)
            if visual_ranked else 0.0
        )

        alpha = _compute_dynamic_alpha(
            avg_transcript_score, avg_visual_score, has_semantic, video_genre
        )
        logger.info(
            "TwoStageFusion: alpha=%.3f genre=%s transcript_chunks=%d visual_frames=%d",
            alpha, video_genre, len(transcript_ranked), len(visual_ranked),
        )

        fused_results = self._weighted_fusion(
            transcript_ranked, visual_ranked, alpha,
            videos, stage1_chunks, has_semantic,
        )

        ocr_ranked = self._build_ocr_ranking(frames, ocr, videos, query_tokens) if has_text else []
        if ocr_ranked:
            fused_results = self._apply_ocr_boost(fused_results, ocr_ranked)

        if len(query_groups) > 1:
            fused_results = self._apply_temporal_boost(fused_results, query_groups)

        fused_results.sort(key=lambda x: x["confidence"], reverse=True)
        return fused_results

    def _weighted_fusion(
        self,
        transcript_ranked: list[dict],
        visual_ranked: list[dict],
        alpha: float,
        videos: dict,
        stage1_chunks: list[dict],
        has_semantic: bool,
    ) -> list[dict]:
        frames_by_video: dict[str, list[dict]] = {}
        for chunk in stage1_chunks:
            vid = chunk["video_id"]
            frames_by_video.setdefault(vid, [])

        for item in visual_ranked:
            vid = item["video_id"]
            frames_by_video.setdefault(vid, []).append(item)

        scored_by_frame: dict[str, dict] = {}

        for item in transcript_ranked:
            vid = item["video_id"]
            t_score = item["_score"]
            mid_ms = item.get("timestamp_ms", 0)

            visual_candidates = frames_by_video.get(vid, [])
            if visual_candidates:
                best_visual = max(visual_candidates, key=lambda x: x["_score"])
                v_score = best_visual["_score"]
            else:
                v_score = 0.0

            if has_semantic:
                confidence = alpha * t_score + (1 - alpha) * v_score
            else:
                confidence = t_score

            frame_id = item.get("frame_id", "")
            if not frame_id:
                frame_part = f"{int(mid_ms / 1000 * 25):06d}"
                frame_id = f"{vid}_{frame_part}"

            image_url = item.get("image_url", "")
            if not image_url:
                image_url = _frame_image_url(vid, frame_id)

            scored_by_frame[frame_id] = {
                "video_id":        vid,
                "youtube_id":      str(videos.get(vid, {}).get("youtube_id", "")),
                "frame_id":        frame_id,
                "frame_number":    item.get("frame_number", 0),
                "timestamp_ms":    mid_ms,
                "confidence":      round(max(0.0, min(1.0, confidence)), 4),
                "frame_image_url": image_url,
                "fps":             float(videos.get(vid, {}).get("fps", 25.0)),
                "_transcript_score": t_score,
                "_visual_score":   v_score,
            }

        for item in visual_ranked:
            fid = item.get("frame_id", "")
            if not fid or fid in scored_by_frame:
                continue
            vid = item["video_id"]
            v_score = item["_score"]

            if has_semantic:
                confidence = (1 - alpha) * v_score
            else:
                confidence = v_score

            scored_by_frame[fid] = {
                "video_id":        vid,
                "youtube_id":      str(videos.get(vid, {}).get("youtube_id", "")),
                "frame_id":        fid,
                "frame_number":    item.get("frame_number", 0),
                "timestamp_ms":    item.get("timestamp_ms", 0),
                "confidence":      round(max(0.0, min(1.0, confidence)), 4),
                "frame_image_url": _frame_image_url(vid, fid, item.get("image_url")),
                "fps":             float(videos.get(vid, {}).get("fps", 25.0)),
                "_transcript_score": 0.0,
                "_visual_score":   v_score,
            }

        return list(scored_by_frame.values())

    def _build_transcript_ranking(
        self, chunks: list[dict], videos: dict,
    ) -> list[dict]:
        scored_by_video: dict[str, dict] = {}
        for chunk in chunks:
            vid = chunk["video_id"]
            score = float(chunk.get("score", 0.0))
            if vid not in scored_by_video or score > scored_by_video[vid]["_score"]:
                mid_ms = (int(chunk["start_time_ms"]) + int(chunk["end_time_ms"])) // 2
                scored_by_video[vid] = {
                    "video_id":      vid,
                    "start_time_ms": chunk["start_time_ms"],
                    "end_time_ms":   chunk["end_time_ms"],
                    "timestamp_ms":  mid_ms,
                    "text":          chunk.get("text", ""),
                    "frame_id":      chunk.get("frame_id", ""),
                    "frame_number":  chunk.get("frame_number", 0),
                    "image_url":     chunk.get("frame_image_url", ""),
                    "_score":        score,
                }
        ranked = sorted(scored_by_video.values(), key=lambda x: x["_score"], reverse=True)
        return ranked

    def _build_visual_ranking(
        self, frames: list[dict], videos: dict, has_semantic: bool,
    ) -> list[dict]:
        scored = []
        seen_frames: set[str] = set()
        for frame in frames:
            fid = frame.get("frame_id", "")
            if fid in seen_frames:
                continue
            seen_frames.add(fid)
            score = float(frame.get("score", 0.5))
            if not has_semantic:
                score = 0.0
            scored.append({
                "video_id":     frame["video_id"],
                "frame_id":     fid,
                "frame_number": frame.get("frame_number", 0),
                "timestamp_ms": frame.get("timestamp_ms", 0),
                "image_url":    frame.get("image_url", ""),
                "_score":       score,
            })
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored

    def _build_ocr_ranking(
        self, frames: list[dict], ocr: list[dict], videos: dict, query_tokens: set[str],
    ) -> list[dict]:
        if not query_tokens or not ocr:
            return []
        scored_by_frame: dict[str, dict] = {}
        for item in ocr:
            score = _text_score(item.get("ocr_text", ""), query_tokens)
            if score <= 0:
                continue
            fid = item.get("frame_id", "")
            if fid not in scored_by_frame or score > scored_by_frame[fid]["_score"]:
                scored_by_frame[fid] = {
                    "video_id":     item["video_id"],
                    "frame_id":     fid,
                    "frame_number": item.get("frame_number", 0),
                    "timestamp_ms": item.get("timestamp_ms", 0),
                    "_score":       score,
                }
        ranked = sorted(scored_by_frame.values(), key=lambda x: x["_score"], reverse=True)
        return ranked

    def _apply_ocr_boost(
        self, results: list[dict], ocr_ranked: list[dict],
    ) -> list[dict]:
        ocr_scores = {item["frame_id"]: item["_score"] for item in ocr_ranked}
        for r in results:
            fid = r.get("frame_id", "")
            if fid in ocr_scores:
                boost = ocr_scores[fid] * 0.05
                r["confidence"] = round(min(1.0, r["confidence"] + boost), 4)
        return results

    def _apply_temporal_boost(
        self, results: list[dict], query_groups: list[dict],
    ) -> list[dict]:
        tolerance_ms = 3000
        by_video: dict[str, list[dict]] = {}
        for r in results:
            by_video.setdefault(r["video_id"], []).append(r)
        for video_results in by_video.values():
            video_results.sort(key=lambda x: x["timestamp_ms"])

        offsets = [int(g.get("temporal_offset_ms", 0)) for g in query_groups[1:]]
        boosted = []
        for r in results:
            temporal_bonus = 0.0
            for offset in offsets:
                target = int(r["timestamp_ms"]) + offset
                nearby = [
                    c["confidence"]
                    for c in by_video.get(r["video_id"], [])
                    if c.get("frame_id") != r.get("frame_id")
                    and abs(int(c["timestamp_ms"]) - target) <= tolerance_ms
                ]
                if nearby:
                    temporal_bonus += max(nearby) * 0.05
            r = dict(r)
            r["confidence"] = round(min(1.0, r["confidence"] + temporal_bonus), 4)
            boosted.append(r)
        return boosted

    def _fallback_results(self, frames: list[dict], videos: dict) -> list[dict]:
        results = []
        for frame in frames:
            vid = frame["video_id"]
            fid = frame["frame_id"]
            video = videos.get(vid, {})
            results.append({
                "video_id":        vid,
                "youtube_id":      str(video.get("youtube_id", "")),
                "frame_id":        fid,
                "frame_number":    frame.get("frame_number", 0),
                "timestamp_ms":    frame.get("timestamp_ms", 0),
                "confidence":      0.0,
                "frame_image_url": _frame_image_url(vid, fid, frame.get("image_url")),
                "fps":             float(video.get("fps", 25.0)),
            })
        return results

    def _visual_only_results(
        self, frames: list[dict], videos: dict, has_semantic: bool,
    ) -> list[dict]:
        results = []
        seen: set[str] = set()
        for frame in frames:
            fid = frame.get("frame_id", "")
            if fid in seen:
                continue
            seen.add(fid)
            vid = frame["video_id"]
            video = videos.get(vid, {})
            score = float(frame.get("score", 0.0)) if has_semantic else 0.0
            results.append({
                "video_id":        vid,
                "youtube_id":      str(video.get("youtube_id", "")),
                "frame_id":        fid,
                "frame_number":    frame.get("frame_number", 0),
                "timestamp_ms":    frame.get("timestamp_ms", 0),
                "confidence":      round(max(0.0, min(1.0, score)), 4),
                "frame_image_url": _frame_image_url(vid, fid, frame.get("image_url")),
                "fps":             float(video.get("fps", 25.0)),
            })
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def _transcript_only_results(
        self, chunks: list[dict], frames: list[dict], videos: dict,
    ) -> list[dict]:
        frames_by_video: dict[str, list[dict]] = {}
        for frame in frames:
            frames_by_video.setdefault(frame["video_id"], []).append(frame)

        results = []
        for chunk in chunks:
            vid = chunk["video_id"]
            score = float(chunk.get("score", 0.0))
            mid_ms = (int(chunk["start_time_ms"]) + int(chunk["end_time_ms"])) // 2

            video_frames = frames_by_video.get(vid, [])
            if video_frames:
                nearest = min(video_frames, key=lambda f: abs(int(f["timestamp_ms"]) - mid_ms))
                fid = nearest["frame_id"]
                fnum = nearest.get("frame_number", 0)
                ts = nearest.get("timestamp_ms", mid_ms)
                img = nearest.get("image_url", "")
            else:
                frame_part = f"{int(mid_ms / 1000 * 25):06d}"
                fid = f"{vid}_{frame_part}"
                fnum = int(mid_ms / 1000 * 25)
                ts = mid_ms
                img = ""

            video = videos.get(vid, {})
            results.append({
                "video_id":        vid,
                "youtube_id":      str(video.get("youtube_id", "")),
                "frame_id":        fid,
                "frame_number":    fnum,
                "timestamp_ms":    ts,
                "confidence":      round(max(0.0, min(1.0, score)), 4),
                "frame_image_url": _frame_image_url(vid, fid, img),
                "fps":             float(video.get("fps", 25.0)),
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
