from app.strategies.base_strategy import BaseStrategy


def _frame_image_url(video_id: str, frame_id: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    frame_part = frame_id.rsplit("_", 1)[-1]
    return f"/static/frames/{video_id}/{frame_part}.jpg"


def _tokens(text: str) -> set[str]:
    return {part.lower() for part in text.replace("_", " ").split() if len(part.strip()) >= 2}


class TranscriptSearch(BaseStrategy):
    """
    Transcript-first strategy. Scores every transcript interval against the
    text query, maps each match to its nearest keyframe, and returns frame-level
    results ranked by transcript relevance.

    Works with MOCK, SAMPLE, and SERVER/LOCAL data.
    When no text query is provided, falls back to visual scoring via frame score.
    """

    name = "Transcript Search v1"
    description = "Searches transcripts first, maps matches to nearest keyframes by timestamp."
    author = "Team AIC 2026"
    version = "1.0"

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        transcripts = raw_data.get("transcripts", [])
        videos = raw_data.get("videos", {})

        text_query = " ".join(g.get("text_query", "") for g in query_groups).strip()
        has_semantic = any(g.get("semantic_query", "").strip() for g in query_groups)

        # Build lookup structures
        frames_by_video: dict[str, list[dict]] = {}
        for frame in frames:
            frames_by_video.setdefault(frame["video_id"], []).append(frame)

        # Score transcripts
        transcript_matches: list[dict] = []
        if text_query:
            query_tokens = _tokens(text_query)
            for t in transcripts:
                score = self._token_match_score(t.get("text", ""), query_tokens)
                if score > 0:
                    transcript_matches.append({**t, "_score": score})

        # Map each transcript match to its nearest frame
        scored_frames: dict[str, dict] = {}  # keyed by frame_id
        for tmatch in transcript_matches:
            vid = tmatch["video_id"]
            mid_ms = (int(tmatch["start_time_ms"]) + int(tmatch["end_time_ms"])) // 2
            nearest = self._nearest_frame(frames_by_video.get(vid, []), mid_ms)
            if nearest is None:
                continue

            fid = nearest["frame_id"]
            transcript_score = float(tmatch["_score"])
            visual_bonus = float(nearest.get("score", 0.5)) * 0.1

            if fid in scored_frames:
                # Keep the max transcript score for this frame
                scored_frames[fid]["confidence"] = max(
                    scored_frames[fid]["confidence"],
                    round(min(1.0, transcript_score + visual_bonus), 4),
                )
            else:
                video = videos.get(vid, {})
                scored_frames[fid] = {
                    "video_id":        nearest["video_id"],
                    "youtube_id":      str(video.get("youtube_id") or ""),
                    "frame_id":        fid,
                    "frame_number":    nearest["frame_number"],
                    "timestamp_ms":    nearest["timestamp_ms"],
                    "confidence":      round(min(1.0, transcript_score + visual_bonus), 4),
                    "frame_image_url": _frame_image_url(
                        nearest["video_id"], fid, nearest.get("image_url")
                    ),
                    "fps":             float(video.get("fps", 25.0)),
                }

        # If no transcript matches and no semantic query, fall back to all frames with neutral score
        if not scored_frames:
            for frame in frames:
                if has_semantic:
                    visual_score = float(frame.get("score", 0.5))
                else:
                    visual_score = 0.0
                fid = frame["frame_id"]
                video = videos.get(frame["video_id"], {})
                scored_frames[fid] = {
                    "video_id":        frame["video_id"],
                    "youtube_id":      str(video.get("youtube_id") or ""),
                    "frame_id":        fid,
                    "frame_number":    frame["frame_number"],
                    "timestamp_ms":    frame["timestamp_ms"],
                    "confidence":      round(max(0.0, min(1.0, visual_score)), 4),
                    "frame_image_url": _frame_image_url(
                        frame["video_id"], fid, frame.get("image_url")
                    ),
                    "fps":             float(video.get("fps", 25.0)),
                }

        results = list(scored_frames.values())
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    @staticmethod
    def _token_match_score(text: str, query_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0
        haystack = text.lower()
        matched = sum(1 for t in query_tokens if t in haystack)
        return matched / len(query_tokens)

    @staticmethod
    def _nearest_frame(frames: list[dict], target_ms: int) -> dict | None:
        if not frames:
            return None
        return min(frames, key=lambda f: abs(int(f["timestamp_ms"]) - target_ms))
