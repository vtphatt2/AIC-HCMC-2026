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

# RRF here fuses independent single-frame candidates, not temporal chains —
# duplicate-filtering collides them much more often than it collides
# temporal's chains. A chain only counts as a duplicate of another chain
# when *every* corresponding step is near-identical simultaneously (rare);
# a single frame collides with any one near-duplicate neighbor directly
# (common — the true match's own neighboring keyframes, from the same
# scene, are exactly the kind of near-duplicate this filter is designed to
# collapse). SearchContext's own OVERSAMPLE_FACTOR (1.5x, first page only)
# isn't enough headroom for that higher collision rate, so oversample
# harder specifically here rather than raising it for every strategy that
# shares base_strategy.py. There's no DP here (unlike temporal) to pay for,
# so the extra width is comparatively cheap.
PER_DETAIL_LIMIT = PER_QUERY_LIMIT * 4


async def run_multi_detail(context):
    groups = [
        group for group in context.query_groups
        if str(group.get("query", "")).strip()
    ]
    if not groups:
        raise ValueError("Multi-detail search requires at least one query")

    # Keep the full candidate rankings; only defer the vectors used by dedup.
    # Larger outputs/genre filters keep the original path (benchmarked slower
    # or sensitive to retrieval ordering when vectors were fetched separately).
    vector_options = {"include_vector": False} if (
        context.top_k <= 100 and context.video_genre in {"", "All"}
    ) else {}
    rankings = await asyncio.gather(*[
        context.retrieve(
            "raw.semantic",
            group["query"].strip(),
            top_k=max(context.top_k, PER_DETAIL_LIMIT),
            **vector_options,
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
