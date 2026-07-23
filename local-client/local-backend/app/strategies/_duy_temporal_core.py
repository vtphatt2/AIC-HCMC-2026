"""Shared retrieval and DP matching for the Duy temporal strategies."""
import asyncio


PER_QUERY_LIMIT = 300
TEMPORAL_CONFIG_SCHEMA = {
    "event_weights": {
        "type": "number_list",
        "label": "Event weights",
        "item_label": "Event {index}",
        "default": [1.0],
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "min_items": 1,
        "max_items": 20,
    }
}


def resolve_event_weights(values, count):
    values = values if isinstance(values, list) else []
    return [float(values[index]) if index < len(values) else 1.0 for index in range(count)]


async def run_temporal(context, interval_min_ms=None, interval_max_ms=None):
    groups = [
        group for group in context.query_groups
        if str(group.get("query", "")).strip()
    ]
    if not groups:
        raise ValueError("Temporal search requires at least one query")

    rankings = await asyncio.gather(*[
        context.retrieve(
            "raw.semantic",
            group["query"].strip(),
            top_k=max(context.top_k, PER_QUERY_LIMIT),
        )
        for group in groups
    ])
    hits = match_temporal(
        rankings,
        groups,
        interval_min_ms,
        interval_max_ms,
        event_weights=context.option("event_weights", None),
    )

    return temporal_results(context, hits)


def temporal_results(context, hits):
    results = []
    for hit in hits:
        row = context.results([hit])[0]
        if len(hit["_steps"]) > 1:
            row["steps"] = context.results([
                {**step, "confidence": hit["confidence"]}
                for step in hit["_steps"]
            ])
        results.append(row)
    return results


def match_temporal(
    rankings, groups, interval_min_ms=None, interval_max_ms=None, *, event_weights=None
):
    """Return the best valid chain ending at each last-level frame."""
    if not rankings:
        return []
    if (interval_min_ms is None) != (interval_max_ms is None):
        raise ValueError("Both interval bounds must be provided")

    levels_by_video = {}
    for level, ranking in enumerate(rankings):
        for hit in ranking:
            levels_by_video.setdefault(hit["video_id"], {}).setdefault(level, []).append(hit)

    offsets = [int(group.get("temporal_offset_ms") or 0) for group in groups]
    weights = resolve_event_weights(event_weights, len(rankings))
    results = []
    for levels in levels_by_video.values():
        if len(levels) != len(rankings):
            continue
        for frames in levels.values():
            frames.sort(key=lambda frame: int(frame["timestamp_ms"]))

        if interval_min_ms is None:
            results.extend(_match_unbounded(levels, offsets, weights))
        else:
            results.extend(_match_interval(
                levels, offsets, int(interval_min_ms), int(interval_max_ms), weights
            ))

    results.sort(key=lambda hit: hit["confidence"], reverse=True)
    return results


def _result(entry, weight_total):
    last = entry["path"][-1]
    return {
        **last,
        "confidence": round(entry["score"] / (weight_total or 1.0), 4),
        "evidence": entry["path"],
        "_steps": entry["path"],
    }


def _match_unbounded(levels, offsets, weights):
    level_count = len(levels)
    dp = [
        {"score": float(frame.get("score", 0.0)) * weights[0], "path": [frame]}
        for frame in levels[0]
    ]
    previous_frames = levels[0]

    for level in range(1, level_count):
        best = None
        pointer = 0
        current_dp = []
        for frame in levels[level]:
            threshold = int(frame["timestamp_ms"]) - offsets[level]
            while (
                pointer < len(previous_frames)
                and int(previous_frames[pointer]["timestamp_ms"]) <= threshold
            ):
                candidate = dp[pointer]
                if candidate is not None and (
                    best is None or candidate["score"] > best["score"]
                ):
                    best = candidate
                pointer += 1
            current_dp.append(None if best is None else {
                "score": best["score"] + float(frame.get("score", 0.0)) * weights[level],
                "path": best["path"] + [frame],
            })
        previous_frames = levels[level]
        dp = current_dp

    return [_result(entry, sum(weights)) for entry in dp if entry is not None]


def _match_interval(levels, offsets, interval_min_ms, interval_max_ms, weights):
    level_count = len(levels)
    if level_count == 1:
        if not interval_min_ms <= 0 <= interval_max_ms:
            return []
        return [
            _result({"score": float(frame.get("score", 0.0)) * weights[0], "path": [frame]}, sum(weights))
            for frame in levels[0]
        ]

    results = []
    last_level = level_count - 1
    for start in levels[0]:
        frontier_frames = [start]
        frontier = [{"score": float(start.get("score", 0.0)) * weights[0], "path": [start]}]

        for level in range(1, last_level):
            best = None
            pointer = 0
            next_frames = []
            next_frontier = []
            for frame in levels[level]:
                threshold = int(frame["timestamp_ms"]) - offsets[level]
                while (
                    pointer < len(frontier_frames)
                    and int(frontier_frames[pointer]["timestamp_ms"]) <= threshold
                ):
                    candidate = frontier[pointer]
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate
                    pointer += 1
                if best is not None:
                    next_frames.append(frame)
                    next_frontier.append({
                        "score": best["score"] + float(frame.get("score", 0.0)) * weights[level],
                        "path": best["path"] + [frame],
                    })
            frontier_frames, frontier = next_frames, next_frontier
            if not frontier_frames:
                break
        if not frontier_frames:
            continue

        best = None
        pointer = 0
        start_time = int(start["timestamp_ms"])
        for end in levels[last_level]:
            span_ms = int(end["timestamp_ms"]) - start_time
            if span_ms > interval_max_ms:
                break
            threshold = int(end["timestamp_ms"]) - offsets[last_level]
            while (
                pointer < len(frontier_frames)
                and int(frontier_frames[pointer]["timestamp_ms"]) <= threshold
            ):
                candidate = frontier[pointer]
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
                pointer += 1
            if span_ms >= interval_min_ms and best is not None:
                results.append(_result({
                    "score": best["score"] + float(end.get("score", 0.0)) * weights[last_level],
                    "path": best["path"] + [end],
                }, sum(weights)))
    return results
