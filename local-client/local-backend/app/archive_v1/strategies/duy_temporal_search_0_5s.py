from app.strategies.base_strategy import BaseStrategy
from app.strategies._temporal_dp_core import run_temporal_interval_dp


class DuyTemporalSearchInterval0to5s(BaseStrategy):
    """
    Same DP as duy_temporal_search.py (v1), plus a bound on the total span:
    last query group's frame time - first query group's frame time must fall
    within [0, 5000] ms.
    """

    name = "Temporal Search v2 (0-5s)"
    description = (
        "DP temporal localization across multiple query groups, restricted to "
        "chains whose first-to-last frame span is within 0-5s."
    )
    author = "Team AIC 2026"
    version = "1.0"

    INTERVAL_MIN_MS = 0
    INTERVAL_MAX_MS = 5_000

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        return run_temporal_interval_dp(raw_data, query_groups, self.INTERVAL_MIN_MS, self.INTERVAL_MAX_MS)
