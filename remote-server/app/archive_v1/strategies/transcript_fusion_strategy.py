from app.strategies.base_strategy import BaseStrategy


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


def _nearest_frame(frames: list[dict], target_ms: int) -> dict | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(int(f["timestamp_ms"]) - target_ms))


def _reciprocal_rank_fusion(
    rankings: list[list[dict]],
    key_fn,
    k: int = 60,
) -> dict:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF_score(d) = sum over lists L: 1 / (k + rank_of_d_in_L)

    Args:
        rankings: list of ranked lists (each already sorted best-first)
        key_fn: function to extract a unique key from each item
        k: RRF constant (default 60)

    Returns:
        dict mapping key → RRF score
    """
    scores: dict = {}
    for ranked_list in rankings:
        for rank, item in enumerate(ranked_list, start=1):
            key = key_fn(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


class TranscriptFusionStrategy(BaseStrategy):
    """
    Fuses PECore visual similarity with topic-based transcript chunk vector
    search using Reciprocal Rank Fusion (RRF).

    When transcript_chunks are available (SERVER/LOCAL mode), they contribute
    equally via RRF. Falls back to token-based transcript scoring when
    transcript_chunks are unavailable (MOCK/SAMPLE mode).
    """

    name = "Transcript Fusion v1"
    description = (
        "RRF fusion of PE-Core visual search and topic-based transcript chunk "
        "vector search. Falls back to token matching when chunk search unavailable."
    )
    author = "Team AIC 2026"
    version = "1.0"

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        ocr = raw_data.get("ocr", [])
        transcripts = raw_data.get("transcripts", [])
        transcript_chunks = raw_data.get("transcript_chunks", [])
        videos = raw_data.get("videos", {})
        video_genre = raw_data.get("video_genre", "All")

        text_query = " ".join(g.get("text_query", "") for g in query_groups).strip()
        has_semantic = any(g.get("semantic_query", "").strip() for g in query_groups)
        has_text = bool(text_query)
        query_tokens = _tokens(text_query)

        # Index frames by video
        frames_by_video: dict[str, list[dict]] = {}
        for frame in frames:
            vid = frame["video_id"]
            frames_by_video.setdefault(vid, []).append(frame)

        # Build visual ranking
        visual_ranked = self._build_visual_ranking(frames, videos, has_semantic)

        # Build transcript ranking
        if transcript_chunks:
            transcript_ranked = self._build_chunk_ranking(
                transcript_chunks, frames_by_video, videos
            )
        else:
            transcript_ranked = self._build_token_transcript_ranking(
                frames, transcripts, frames_by_video, videos, query_tokens
            )

        # Also include OCR ranking if text is provided
        ocr_ranked = self._build_ocr_ranking(frames, ocr, videos, query_tokens) if has_text else []

        # Fuse rankings with RRF
        rankings = [visual_ranked]
        if transcript_ranked:
            rankings.append(transcript_ranked)
        if ocr_ranked and has_text:
            rankings.append(ocr_ranked)

        rrf_scores = _reciprocal_rank_fusion(
            rankings,
            key_fn=lambda item: item["video_id"],
        )

        # Build per-frame results using the nearest frame for each chunk hit
        scored_by_frame: dict[str, dict] = {}

        # Process chunk-ranked items: map to nearest keyframe
        for item in transcript_ranked:
            vid = item["video_id"]
            mid_ms = item.get("timestamp_ms", 0)
            if mid_ms == 0 and "start_time_ms" in item:
                mid_ms = (int(item["start_time_ms"]) + int(item["end_time_ms"])) // 2
            nearest = _nearest_frame(frames_by_video.get(vid, []), mid_ms)
            if nearest is None:
                continue
            fid = nearest["frame_id"]
            rrf_vid = rrf_scores.get(vid, 0.0)
            confidence = self._normalize_rrf(rrf_vid, len(rankings))
            scored_by_frame[fid] = self._make_result(
                nearest, videos, confidence, fid, nearest.get("image_url")
            )

        # Process visual-ranked items
        for item in visual_ranked:
            vid = item["video_id"]
            fid = item.get("frame_id", "")
            if not fid or fid in scored_by_frame:
                continue
            rrf_vid = rrf_scores.get(vid, 0.0)
            confidence = self._normalize_rrf(rrf_vid, len(rankings))
            scored_by_frame[fid] = self._make_result(
                item, videos, confidence, fid, item.get("image_url")
            )

        # Fallback: no semantic, no text, no chunks → return all frames at neutral
        if not scored_by_frame:
            for frame in frames:
                fid = frame["frame_id"]
                video = videos.get(frame["video_id"], {})
                scored_by_frame[fid] = {
                    "video_id":        frame["video_id"],
                    "youtube_id":      str(video.get("youtube_id") or ""),
                    "frame_id":        fid,
                    "frame_number":    frame["frame_number"],
                    "timestamp_ms":    frame["timestamp_ms"],
                    "confidence":      0.0,
                    "frame_image_url": _frame_image_url(
                        frame["video_id"], fid, frame.get("image_url")
                    ),
                    "fps":             float(video.get("fps", 25.0)),
                }

        results = list(scored_by_frame.values())
        results = self._apply_genre_boost(results, video_genre, ocr, transcript_chunks)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def _apply_genre_boost(
        self,
        results: list[dict],
        genre: str,
        ocr: list[dict],
        transcript_chunks: list[dict],
    ) -> list[dict]:
        if genre == "All" or genre not in {
            "Giáo dục", "Ẩm thực", "Thể thao", "Công nghệ", "Pháp luật",
        }:
            return results

        ocr_frame_ids = {r.get("frame_id") for r in ocr}
        chunk_video_ids = {c.get("video_id") for c in transcript_chunks}

        for r in results:
            bonus = 0.0
            fid = r.get("frame_id", "")
            vid = r.get("video_id", "")

            if genre == "Giáo dục" and fid in ocr_frame_ids:
                bonus = 0.05  # slides/onscreen text boost
            elif genre == "Ẩm thực" and r.get("confidence", 0) > 0.6:
                bonus = 0.03  # visual confidence boost for cooking scenes
            elif genre == "Thể thao" and r.get("confidence", 0) > 0.5:
                bonus = 0.02
            elif genre == "Công nghệ" and (fid in ocr_frame_ids or vid in chunk_video_ids):
                bonus = 0.04  # demo/narration combo boost
            elif genre == "Pháp luật" and (fid in ocr_frame_ids):
                bonus = 0.05  # legal documents/text

            r["confidence"] = round(min(1.0, r["confidence"] + bonus), 4)
        return results

    # ── Ranking builders ───────────────────────────────────────────────────

    def _build_visual_ranking(
        self, frames: list[dict], videos: dict, has_semantic: bool
    ) -> list[dict]:
        scored = []
        seen = set()
        for frame in frames:
            vid = frame["video_id"]
            if vid in seen:
                continue
            seen.add(vid)
            score = float(frame.get("score", 0.5))
            if not has_semantic:
                score = 0.0
            scored.append({
                "video_id":     vid,
                "frame_id":     frame.get("frame_id", ""),
                "frame_number": frame.get("frame_number", 0),
                "timestamp_ms": frame.get("timestamp_ms", 0),
                "image_url":    frame.get("image_url", ""),
                "_score":       score,
            })
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored

    def _build_chunk_ranking(
        self,
        chunks: list[dict],
        frames_by_video: dict[str, list[dict]],
        videos: dict,
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
                    "_score":        score,
                }
        ranked = sorted(scored_by_video.values(), key=lambda x: x["_score"], reverse=True)
        return ranked

    def _build_token_transcript_ranking(
        self,
        frames: list[dict],
        transcripts: list[dict],
        frames_by_video: dict[str, list[dict]],
        videos: dict,
        query_tokens: set[str],
    ) -> list[dict]:
        if not query_tokens:
            return []
        scored_by_video: dict[str, dict] = {}
        for t in transcripts:
            score = _text_score(t.get("text", ""), query_tokens)
            if score <= 0:
                continue
            vid = t["video_id"]
            if vid not in scored_by_video or score > scored_by_video[vid]["_score"]:
                mid_ms = (int(t["start_time_ms"]) + int(t["end_time_ms"])) // 2
                scored_by_video[vid] = {
                    "video_id":      vid,
                    "start_time_ms": t["start_time_ms"],
                    "end_time_ms":   t["end_time_ms"],
                    "timestamp_ms":  mid_ms,
                    "_score":        score,
                }
        ranked = sorted(scored_by_video.values(), key=lambda x: x["_score"], reverse=True)
        return ranked

    def _build_ocr_ranking(
        self,
        frames: list[dict],
        ocr: list[dict],
        videos: dict,
        query_tokens: set[str],
    ) -> list[dict]:
        if not query_tokens or not ocr:
            return []
        scored_by_video: dict[str, dict] = {}
        for item in ocr:
            score = _text_score(item.get("ocr_text", ""), query_tokens)
            if score <= 0:
                continue
            vid = item["video_id"]
            if vid not in scored_by_video or score > scored_by_video[vid]["_score"]:
                scored_by_video[vid] = {
                    "video_id":     vid,
                    "frame_id":     item.get("frame_id", ""),
                    "frame_number": item.get("frame_number", 0),
                    "timestamp_ms": item.get("timestamp_ms", 0),
                    "image_url":    item.get("image_url", ""),
                    "_score":       score,
                }
        ranked = sorted(scored_by_video.values(), key=lambda x: x["_score"], reverse=True)
        return ranked

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_rrf(rrf_score: float, num_lists: int) -> float:
        max_possible = num_lists / 61.0  # k=60, rank=1 gives 1/(60+1) per list
        if max_possible <= 0:
            return 0.0
        return round(min(1.0, rrf_score / max_possible), 4)

    @staticmethod
    def _make_result(
        item: dict, videos: dict, confidence: float,
        frame_id: str, image_url: str | None = None
    ) -> dict:
        video = videos.get(item["video_id"], {})
        return {
            "video_id":        item["video_id"],
            "youtube_id":      str(video.get("youtube_id") or ""),
            "frame_id":        frame_id,
            "frame_number":    item.get("frame_number", 0),
            "timestamp_ms":    item.get("timestamp_ms", 0),
            "confidence":      confidence,
            "frame_image_url": _frame_image_url(item["video_id"], frame_id, image_url),
            "fps":             float(video.get("fps", 25.0)),
        }
