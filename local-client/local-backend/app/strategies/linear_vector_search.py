from app.strategies.base_strategy import BaseStrategy


class LinearVectorSearch(BaseStrategy):
    """
    Local SAMPLE-mode baseline: cosine search over precomputed PECore frame vectors.

    Query format for now:
      - semantic_query = "L01_V001_000022"
      - semantic_query = "L01_V001/000022"
      - semantic_query = "L01_V001 000022"

    This is an image-to-image vector search baseline. Text-to-vector encoding comes later.
    """

    name = "Linear PECore Search"
    description = "Linear cosine search over local AIC2026_sample PECore keyframe vectors."
    author = "Team AIC 2026"
    version = "0.1"

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        query = ""
        for group in query_groups:
            query = group.get("semantic_query", "").strip()
            if query:
                break

        frame_id = self.data_provider.resolve_sample_frame_id(query)
        if not frame_id:
            raise ValueError(
                "Linear PECore Search expects semantic_query to be a frame id, "
                "for example: L01_V001_000022"
            )

        videos = raw_data.get("videos", {})
        hits = self.data_provider.linear_search_by_frame_id(frame_id, top_k=1000)

        results = []
        for frame in hits:
            video = videos.get(frame["video_id"], {})
            results.append({
                "video_id":        frame["video_id"],
                "youtube_id":      str(video.get("youtube_id") or ""),
                "frame_id":        frame["frame_id"],
                "frame_number":    frame["frame_number"],
                "timestamp_ms":    frame["timestamp_ms"],
                "confidence":      round(float(frame["score"]), 4),
                "frame_image_url": frame["image_url"],
                "fps":             float(video.get("fps", 25.0)),
            })

        return results
