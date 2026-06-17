import random
from app.strategies.base_strategy import BaseStrategy


class ExampleStrategy(BaseStrategy):
    """
    A simple example strategy for UI testing and onboarding.

    It does NOT do real retrieval — it just shuffles all mock frames,
    assigns random confidence scores, and returns them sorted.

    Use this as a template to write your own strategy.
    Copy this file, rename it (e.g. yourname_idea_v1.py), change the class
    attributes and implement fusion_and_temporal() with your real algorithm.
    """

    name = "Example (Mock) Duy"
    description = (
        "Returns all mock frames with random confidence scores. "
        "Used for UI testing — replace with a real strategy for experiments."
    )
    author = "Team AIC 2026"
    version = "1.0"

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        videos = raw_data.get("videos", {})

        results = []
        for frame in frames:
            video = videos.get(frame["video_id"], {})
            fps = float(video.get("fps", 25.0))

            results.append({
                "video_id":        frame["video_id"],
                "youtube_id":      str(video.get("youtube_id") or ""),
                "frame_id":        frame["frame_id"],
                "frame_number":    frame["frame_number"],
                "timestamp_ms":    frame["timestamp_ms"],
                "confidence":      round(random.uniform(0.4, 1.0), 4),
                "frame_image_url": frame["image_url"],
                "fps":             fps,
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
