import asyncio

from .base_strategy import BaseStrategy
from ._fusion import rrf


SOURCE_CONFIG_SCHEMA = {
    "raw.semantic": {"label": "Raw visual", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
    "subtitled.semantic": {"label": "Subtitled visual", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
    "transcript.lexical": {"label": "Transcript BM25", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
    "transcript.semantic": {"label": "Transcript semantic", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
}


async def frames_for_chunks(context, chunks):
    output = []
    for chunk in chunks[:30]:
        frames = await context.keyframes(
            chunk["video_id"], chunk["start_time_ms"], chunk["end_time_ms"], limit=3
        )
        output.extend({
            **frame,
            "channel": chunk["channel"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "evidence": [chunk],
        } for frame in frames)
    return output


async def fused_ranking(context, query):
    async def safe(channel):
        try:
            return await context.retrieve(channel, query, top_k=max(context.top_k, 100))
        except (RuntimeError, ValueError):
            return []

    raw, subtitled, lexical, semantic = await asyncio.gather(*[
        safe(channel) for channel in SOURCE_CONFIG_SCHEMA
    ])
    lexical_frames, semantic_frames = await asyncio.gather(
        frames_for_chunks(context, lexical),
        frames_for_chunks(context, semantic),
    )
    return rrf(
        [raw, subtitled, lexical_frames, semantic_frames],
        key="frame_id",
        weights=[context.option(channel, 1.0) for channel in SOURCE_CONFIG_SCHEMA],
    )


class MultiSource(BaseStrategy):
    name = "Multi-source"
    description = "Fuse raw, subtitled and transcript retrieval with RRF."
    author = "AIC HCMC"
    version = "2.0"
    config_schema = SOURCE_CONFIG_SCHEMA

    async def run(self, context):
        query = next(
            (str(group.get("query", "")).strip() for group in context.query_groups if group.get("query")),
            "",
        )
        if not query:
            raise ValueError("Multi-source requires a non-empty query")

        return context.results(await fused_ranking(context, query))
