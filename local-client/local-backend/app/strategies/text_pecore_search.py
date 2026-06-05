from app.services.text_encoder import PECoreTextEncoder
from app.strategies.base_strategy import BaseStrategy


class TextPECoreSearch(BaseStrategy):
    """
    Local SAMPLE-mode text-to-image search over PECore keyframe vectors.
    """

    name = "Text PECore Search"
    description = "Encode semantic text with PECore/OpenCLIP and linear-search local keyframe vectors."
    author = "Team AIC 2026"
    version = "0.1"

    def __init__(self, data_provider):
        super().__init__(data_provider)
        self.text_encoder = PECoreTextEncoder()

    def pre_process(self, query_groups: list[dict]) -> list[dict]:
        query = self._first_semantic_query(query_groups)
        if not query:
            raise ValueError("Text PECore Search requires a non-empty semantic query")

        query_vector = self.text_encoder.encode(query)
        processed = [dict(group) for group in query_groups]
        processed[0]["_pecore_text_vector"] = query_vector
        return processed

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        query_vector = query_groups[0].get("_pecore_text_vector")
        if query_vector is None:
            raise ValueError("Text PECore Search did not receive an encoded text vector")

        videos = raw_data.get("videos", {})
        hits = self.data_provider.linear_search_by_vector(query_vector, top_k=1000)

        results = []
        for frame in hits:
            video = videos.get(frame["video_id"], {})
            results.append({
                "video_id":        frame["video_id"],
                "frame_id":        frame["frame_id"],
                "frame_number":    frame["frame_number"],
                "timestamp_ms":    frame["timestamp_ms"],
                "confidence":      round(float(frame["score"]), 4),
                "frame_image_url": frame["image_url"],
                "fps":             float(video.get("fps", 25.0)),
            })
        return results

    @staticmethod
    def _first_semantic_query(query_groups: list[dict]) -> str:
        for group in query_groups:
            query = group.get("semantic_query", "").strip()
            if query:
                return query
        return ""
