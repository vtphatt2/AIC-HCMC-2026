import asyncio

from .base_strategy import BaseStrategy
from .multi_source import SOURCE_CONFIG_SCHEMA, fused_ranking
from ._duy_temporal_core import (
    TEMPORAL_CONFIG_SCHEMA,
    match_temporal,
    temporal_results,
)


class MultiSourceTemporal(BaseStrategy):
    name = "Multi-source temporal"
    description = "Fuse four retrieval channels per event, then match events in time."
    author = "AIC HCMC"
    version = "2.0"
    config_schema = {**SOURCE_CONFIG_SCHEMA, **TEMPORAL_CONFIG_SCHEMA}

    async def run(self, context):
        groups = [
            group for group in context.query_groups
            if str(group.get("query", "")).strip()
        ]
        if not groups:
            raise ValueError("Multi-source temporal requires at least one query")

        rankings = await asyncio.gather(*[
            fused_ranking(context, group["query"].strip()) for group in groups
        ])
        rankings = [
            [{**hit, "score": hit["confidence"]} for hit in ranking]
            for ranking in rankings
        ]
        hits = match_temporal(
            rankings,
            groups,
            event_weights=context.option("event_weights", None),
        )
        return temporal_results(context, hits)
