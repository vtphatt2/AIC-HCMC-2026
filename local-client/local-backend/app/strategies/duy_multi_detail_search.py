"""One frame, described across several detail prompts instead of one long
one — each query group's text embeds separately (working around the text
encoder's ~72-token limit for a frame too detailed to fit in one query),
and a frame ranking well across more of them wins via RRF. No time
chaining (unlike duy_temporal_search): every detail is independent, all
searched in parallel for the same single frame."""
import asyncio

from ._duy_temporal_core import PER_QUERY_LIMIT, TEMPORAL_CONFIG_SCHEMA, resolve_event_weights
from ._fusion import rrf
from .base_strategy import BaseStrategy


async def run_multi_detail(context):
    groups = [
        group for group in context.query_groups
        if str(group.get("query", "")).strip()
    ]
    if not groups:
        raise ValueError("Multi-detail search requires at least one query")

    rankings = await asyncio.gather(*[
        context.retrieve(
            "raw.semantic",
            group["query"].strip(),
            top_k=max(context.top_k, PER_QUERY_LIMIT),
        )
        for group in groups
    ])

    weights = resolve_event_weights(context.option("event_weights", None), len(rankings))
    fused = rrf(rankings, key="frame_id", weights=weights)
    return context.results(fused[:context.top_k])


class DuyMultiDetailSearch(BaseStrategy):
    name = "Duy multi-detail search"
    description = "One frame described across several detail prompts — frames matching more prompts rank higher (RRF)."
    author = "Team AIC 2026"
    version = "1.0"
    config_schema = TEMPORAL_CONFIG_SCHEMA

    async def run(self, context):
        return await run_multi_detail(context)
