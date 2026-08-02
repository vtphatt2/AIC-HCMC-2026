from .base_strategy import BaseStrategy


class RawVisual(BaseStrategy):
    name = "Raw visual"
    description = "PE-Core semantic search over raw keyframes."
    author = "AIC HCMC"
    version = "2.0"

    async def run(self, context):
        query = next(
            (str(group.get("query", "")).strip() for group in context.query_groups if group.get("query")),
            "",
        )
        if not query:
            raise ValueError("Raw visual requires a non-empty query")
        return context.results(await context.retrieve("raw.semantic", query))
