from app.strategies.base_strategy import BaseStrategy


class StableFusion(BaseStrategy):
    """
    Production-stable strategy. This file is the reference implementation —
    copy your tested local strategy here once it outperforms this baseline.
    """

    name = "Stable Fusion v1"
    description = (
        "Baseline fusion strategy: ranks frames by OCR text match score, "
        "then applies a simple temporal window to group nearby frames."
    )
    author = "Team AIC 2026"
    version = "1.0"

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        ocr = raw_data.get("ocr", [])
        videos = raw_data.get("videos", {})

        # Build a lookup: frame_id → OCR score (1.0 if present, 0 otherwise)
        ocr_hit_ids = {record["frame_id"] for record in ocr}

        results = []
        seen_frame_ids = set()

        # Score OCR hits first (highest priority)
        for record in ocr:
            fid = record["frame_id"]
            if fid in seen_frame_ids:
                continue
            seen_frame_ids.add(fid)
            video = videos.get(record["video_id"], {})
            results.append({
                "video_id":        record["video_id"],
                "frame_id":        fid,
                "frame_number":    record["frame_number"],
                "timestamp_ms":    record["timestamp_ms"],
                "confidence":      1.0,
                "frame_image_url": f"/static/frames/{record['video_id']}/{fid}.jpg",
                "fps":             float(video.get("fps", 25.0)),
            })

        # Add visual hits that didn't appear in OCR results
        for frame in frames:
            fid = frame["frame_id"]
            if fid in seen_frame_ids:
                continue
            seen_frame_ids.add(fid)
            video = videos.get(frame["video_id"], {})
            results.append({
                "video_id":        frame["video_id"],
                "frame_id":        fid,
                "frame_number":    frame["frame_number"],
                "timestamp_ms":    frame["timestamp_ms"],
                "confidence":      round(frame.get("score", 0.5), 4),
                "frame_image_url": f"/static/frames/{frame['video_id']}/{fid}.jpg",
                "fps":             float(video.get("fps", 25.0)),
            })

        # Sort descending by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
