import os
from app.db import milvus_client, postgres_client

ENV_MODE = os.getenv("ENV_MODE", "SERVER")


class DataProvider:
    """
    SERVER mode: reads directly from local Milvus and PostgreSQL using native SDKs.
    This gives maximum speed and zero network latency during competition.

    The interface is identical to the local-client DataProvider so strategy files
    can be copied to this server without changing a single line.
    """

    def __init__(self):
        if ENV_MODE != "SERVER":
            raise RuntimeError(
                f"remote-server DataProvider only supports ENV_MODE=SERVER, got '{ENV_MODE}'"
            )
        self._collection = milvus_client.get_collection()

    async def get_raw_data(self, query_groups: list[dict], limit: int = 1000) -> dict:
        """
        Fetch multi-modal raw data for all query groups from local databases.
        Returns a unified dict that strategies receive as `raw_data`.
        """
        all_frame_ids: set[str] = set()
        all_video_ids: set[str] = set()
        frames: list[dict] = []
        ocr: list[dict] = []
        transcripts: list[dict] = []

        for group in query_groups:
            semantic_query = group.get("semantic_query", "").strip()
            text_query = group.get("text_query", "").strip()

            # Visual search — requires an embedding vector; placeholder shows the pattern.
            # In production, embed semantic_query with PE-Core-bigG-14-448 first.
            if semantic_query:
                # TODO: replace with actual model inference
                # query_vector = embed_text(semantic_query)
                # hits = milvus_client.vector_search(self._collection, query_vector, top_k=limit)
                # frames.extend(hits)
                pass

            # OCR / transcript text search
            if text_query:
                ocr_hits = await postgres_client.search_ocr_text(text_query, limit=limit)
                ocr.extend(ocr_hits)
                all_frame_ids.update(h["frame_id"] for h in ocr_hits)
                all_video_ids.update(h["video_id"] for h in ocr_hits)

                transcript_hits = await postgres_client.search_transcript_text(text_query, limit=limit)
                transcripts.extend(transcript_hits)
                all_video_ids.update(h["video_id"] for h in transcript_hits)

        # Fetch video metadata for all referenced videos
        video_rows = await postgres_client.fetch_video_metadata(list(all_video_ids))
        videos = {v["video_id"]: dict(v) for v in video_rows}

        # Cap to limit
        return {
            "frames":      frames[:limit],
            "ocr":         ocr[:limit],
            "transcripts": transcripts[:limit],
            "videos":      videos,
        }
