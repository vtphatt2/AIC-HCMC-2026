from ._duy_temporal_core import run_temporal
from .base_strategy import BaseStrategy


class DuyTemporalSearch(BaseStrategy):
    name = "Duy temporal search"
    description = "DP sequence search with each query group's minimum time offset."
    author = "Team AIC 2026"
    version = "2.0"

    async def run(self, context):
        return await run_temporal(context)
