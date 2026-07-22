import asyncio

from .base_strategy import BaseStrategy


class TemporalVisual(BaseStrategy):
    name = "Temporal visual"
    description = "Match raw visual queries in order using each group's time offset."
    author = "AIC HCMC"
    version = "2.0"

    PER_STEP = 300
    TOLERANCE_MS = 3000

    async def run(self, context):
        groups = [group for group in context.query_groups if str(group.get("query", "")).strip()]
        if not groups:
            raise ValueError("Temporal visual requires at least one query")
        rankings = await asyncio.gather(*[
            context.retrieve(
                "raw.semantic",
                group["query"].strip(),
                top_k=max(context.top_k, self.PER_STEP),
            )
            for group in groups
        ])
        if len(rankings) == 1:
            return context.results(rankings[0])

        offsets = [0]
        total = 0
        for group in groups[1:]:
            total += int(group.get("temporal_offset_ms") or 0)
            offsets.append(total)

        matches = []
        for anchor in rankings[0]:
            chain = [anchor]
            anchor_time = int(anchor["timestamp_ms"])
            for step, ranking in enumerate(rankings[1:], start=1):
                expected = anchor_time + offsets[step]
                candidates = [
                    hit for hit in ranking
                    if hit["video_id"] == anchor["video_id"]
                    and abs(int(hit["timestamp_ms"]) - expected) <= self.TOLERANCE_MS
                ]
                if not candidates:
                    break
                chain.append(max(
                    candidates,
                    key=lambda hit: float(hit.get("score", 0.0))
                    - abs(int(hit["timestamp_ms"]) - expected) / self.TOLERANCE_MS,
                ))
            if len(chain) == len(rankings):
                score = sum(float(hit.get("score", 0.0)) for hit in chain) / len(chain)
                matches.append({**chain[-1], "confidence": score, "evidence": chain, "_steps": chain})

        matches.sort(key=lambda hit: hit["confidence"], reverse=True)
        results = context.results(matches)
        for result, hit in zip(results, matches):
            result["steps"] = context.results(hit["_steps"])
            result.pop("_steps", None)
        return results
