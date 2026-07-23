from ._duy_temporal_core import TEMPORAL_CONFIG_SCHEMA, run_temporal
from .base_strategy import BaseStrategy


class DuyTemporalSearch0To5s(BaseStrategy):
    name = "Duy temporal search (0-5s)"
    description = "DP sequence search whose first-to-last span is 0-5 seconds."
    author = "Team AIC 2026"
    version = "2.0"
    config_schema = TEMPORAL_CONFIG_SCHEMA

    async def run(self, context):
        return await run_temporal(context, 0, 5_000)
