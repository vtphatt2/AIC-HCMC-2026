from app.strategies.base_strategy import BaseStrategy


def _frame_image_url(video_id: str, frame_id: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    frame_part = frame_id.rsplit("_", 1)[-1]
    return f"/static/frames/{video_id}/{frame_part}.jpg"


class StableFusion(BaseStrategy):
    """
    Production-stable strategy. This file is the reference implementation —
    copy your tested local strategy here once it outperforms this baseline.
    """

    name = "Stable Fusion v1"
    description = (
        "Fuses PECore visual similarity, OCR keyword matches, transcript overlap, "
        "and simple temporal-step consistency."
    )
    author = "Team AIC 2026"
    version = "1.0"

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        ocr = raw_data.get("ocr", [])
        transcripts = raw_data.get("transcripts", [])
        videos = raw_data.get("videos", {})

        text_query = " ".join(g.get("text_query", "") for g in query_groups)
        semantic_query_present = any(g.get("semantic_query", "").strip() for g in query_groups)
        query_tokens = _tokens(text_query)

        ocr_by_frame = {record["frame_id"]: record for record in ocr}
        transcripts_by_video: dict[str, list[dict]] = {}
        for record in transcripts:
            transcripts_by_video.setdefault(record["video_id"], []).append(record)

        frame_by_id = {}
        for frame in frames:
            frame_by_id[frame["frame_id"]] = dict(frame)
        for record in ocr:
            frame_by_id.setdefault(record["frame_id"], {
                "frame_id": record["frame_id"],
                "video_id": record["video_id"],
                "frame_number": record["frame_number"],
                "timestamp_ms": record["timestamp_ms"],
                "image_url": record.get("image_url"),
            })

        scored = []
        for frame in frame_by_id.values():
            fid = frame["frame_id"]
            video = videos.get(frame["video_id"], {})
            visual_score = _visual_score(frame.get("score"), semantic_query_present)
            ocr_score = _text_score(ocr_by_frame.get(fid, {}).get("ocr_text", ""), query_tokens)
            transcript_score = _transcript_score(frame, transcripts_by_video, query_tokens)
            confidence = _combine_scores(visual_score, ocr_score, transcript_score, semantic_query_present, bool(query_tokens))

            scored.append({
                "video_id":        frame["video_id"],
                "frame_id":        fid,
                "frame_number":    frame["frame_number"],
                "timestamp_ms":    frame["timestamp_ms"],
                "confidence":      confidence,
                "frame_image_url": _frame_image_url(frame["video_id"], fid, frame.get("image_url")),
                "fps":             float(video.get("fps", 25.0)),
            })

        if len(query_groups) > 1:
            scored = _apply_temporal_boost(scored, query_groups)

        scored.sort(key=lambda x: x["confidence"], reverse=True)
        return scored


def _tokens(text: str) -> set[str]:
    return {part.lower() for part in text.replace("_", " ").split() if len(part.strip()) >= 2}


def _visual_score(score, semantic_query_present: bool) -> float:
    if score is None:
        return 0.0 if semantic_query_present else 0.5
    score = float(score)
    if score < 0.0:
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    return max(0.0, min(1.0, score))


def _text_score(text: str, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    haystack = text.lower()
    matched = sum(1 for token in query_tokens if token in haystack)
    return matched / len(query_tokens)


def _transcript_score(frame: dict, transcripts_by_video: dict[str, list[dict]], query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    frame_ms = int(frame["timestamp_ms"])
    best = 0.0
    for transcript in transcripts_by_video.get(frame["video_id"], []):
        if int(transcript["start_time_ms"]) <= frame_ms <= int(transcript["end_time_ms"]):
            best = max(best, _text_score(transcript.get("text", ""), query_tokens))
    return best


def _combine_scores(
    visual_score: float,
    ocr_score: float,
    transcript_score: float,
    has_semantic: bool,
    has_text: bool,
) -> float:
    if has_semantic and has_text:
        score = 0.65 * visual_score + 0.25 * ocr_score + 0.10 * transcript_score
    elif has_semantic:
        score = visual_score
    elif has_text:
        score = 0.75 * ocr_score + 0.25 * transcript_score
    else:
        score = visual_score
    return round(max(0.0, min(1.0, score)), 4)


def _apply_temporal_boost(results: list[dict], query_groups: list[dict]) -> list[dict]:
    tolerance_ms = 3000
    by_video: dict[str, list[dict]] = {}
    for result in results:
        by_video.setdefault(result["video_id"], []).append(result)
    for video_results in by_video.values():
        video_results.sort(key=lambda item: item["timestamp_ms"])

    boosted = []
    offsets = [int(g.get("temporal_offset_ms", 0)) for g in query_groups[1:]]
    for result in results:
        temporal_bonus = 0.0
        video_results = by_video.get(result["video_id"], [])
        for offset in offsets:
            target = int(result["timestamp_ms"]) + offset
            nearby = [
                candidate["confidence"]
                for candidate in video_results
                if abs(int(candidate["timestamp_ms"]) - target) <= tolerance_ms
            ]
            if nearby:
                temporal_bonus += max(nearby) * 0.05
        boosted_result = dict(result)
        boosted_result["confidence"] = round(min(1.0, result["confidence"] + temporal_bonus), 4)
        boosted.append(boosted_result)
    return boosted
