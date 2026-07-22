import asyncio

from .base_strategy import BaseStrategy
from ._fusion import rrf


class MultiSource(BaseStrategy):
    name = "Multi-source"
    description = "Fuse raw, subtitled and transcript retrieval with RRF."
    author = "AIC HCMC"
    version = "2.0"

    async def run(self, context):
        query = next(
            (str(group.get("query", "")).strip() for group in context.query_groups if group.get("query")),
            "",
        )
        if not query:
            raise ValueError("Multi-source requires a non-empty query")

        async def safe(channel):
            try:
                return await context.retrieve(channel, query, top_k=max(context.top_k, 100))
            except (RuntimeError, ValueError):
                return []

        raw, subtitled, lexical, semantic = await asyncio.gather(*[
            safe("raw.semantic"),
            safe("subtitled.semantic"),
            safe("transcript.lexical"),
            safe("transcript.semantic"),
        ])
        lexical_frames, semantic_frames = await asyncio.gather(
            self._frames_for_chunks(context, lexical),
            self._frames_for_chunks(context, semantic),
        )
        return context.results(rrf(
            [raw, subtitled, lexical_frames, semantic_frames],
            key="frame_id",
        ))

    @staticmethod
    async def _frames_for_chunks(context, chunks):
        output = []
        for chunk in chunks[:30]:
            frames = await context.keyframes(
                chunk["video_id"],
                chunk["start_time_ms"],
                chunk["end_time_ms"],
                limit=3,
            )
            output.extend({
                **frame,
                "channel": chunk["channel"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "evidence": [chunk],
            } for frame in frames)
        return output
