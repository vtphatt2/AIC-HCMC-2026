from ._duy_temporal_core import run_temporal
from .base_strategy import BaseStrategy


class DuyTemporalSearch10To20s(BaseStrategy):
    name = "Duy temporal search (10-20s)"
    description = "DP sequence search whose first-to-last span is 10-20 seconds."
    author = "Team AIC 2026"
    version = "2.0"

    async def run(self, context):
        return await run_temporal(context, 10_000, 20_000)
