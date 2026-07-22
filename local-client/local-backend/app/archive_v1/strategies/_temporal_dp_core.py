"""
Shared DP core for temporal-search strategies that bound the total span
(last query group's frame time - first query group's frame time) to an
interval [interval_min_ms, interval_max_ms].

Approach: fix the start frame (level 0) and end frame (last level) with two
loops, then solve the middle levels with the same offset-DP as
duy_temporal_search.py (v1), seeded by the fixed start and closed off by the
fixed end. The middle chain only depends on `start`, not `end`, so it's
computed once per start and reused — for each start, the `end` loop then
just runs the v1 two-pointer scan to find the best middle-chain predecessor
for every end candidate (both lists are sorted by timestamp_ms, so both the
interval-span check and the offset check can early-exit / advance
monotonically instead of rescanning).

Underscore-prefixed filename so main.py's discover_strategies() (which
scans app/strategies/*.py) skips this module — it has no name/description/
author and is not meant to be instantiated directly.
"""


def _make_result(
    video_id: str, youtube_id: str, fps: float, frame: dict, confidence: float,
    path: list[dict] | None = None,
) -> dict:
    rounded = round(confidence, 4)

    def _row(f: dict) -> dict:
        return {
            "video_id":        video_id,
            "youtube_id":      youtube_id,
            "frame_id":        f["frame_id"],
            "frame_number":    f["frame_number"],
            "timestamp_ms":    f["timestamp_ms"],
            "confidence":      rounded,
            "frame_image_url": f["image_url"],
            "fps":             fps,
        }

    result = _row(frame)
    if path is not None and len(path) > 1:
        # Surface every frame in the matched chain, not just the closing one,
        # so the frontend can render the full temporal match as one cluster.
        result["steps"] = [_row(f) for f in path]
    return result


def run_temporal_interval_dp(
    raw_data: dict,
    query_groups: list[dict],
    interval_min_ms: int,
    interval_max_ms: int,
) -> list[dict]:
    frames = raw_data.get("frames", [])
    videos = raw_data.get("videos", {})

    # Only groups that actually ran a semantic search produced candidates.
    effective_indices = [
        i for i, g in enumerate(query_groups)
        if str(g.get("semantic_query", "")).strip()
    ]
    if not effective_indices:
        return []

    offsets = [int(query_groups[i].get("temporal_offset_ms", 0)) for i in effective_indices]
    num_levels = len(effective_indices)

    # frames_by_video[video_id][level] = candidate frames for that query group
    frames_by_video: dict[str, dict[int, list[dict]]] = {}
    for frame in frames:
        g_idx = frame.get("_query_group_index")
        if g_idx not in effective_indices:
            continue
        level = effective_indices.index(g_idx)
        frames_by_video.setdefault(frame["video_id"], {}).setdefault(level, []).append(frame)

    results = []

    for video_id, levels in frames_by_video.items():
        # A complete chain needs at least one candidate at every level.
        if len(levels) != num_levels:
            continue

        for lvl_frames in levels.values():
            lvl_frames.sort(key=lambda f: f["timestamp_ms"])

        video = videos.get(video_id, {})
        fps = float(video.get("fps", 25.0))
        youtube_id = str(video.get("youtube_id") or "")

        # Single query group: no start/end pair, span is always 0.
        if num_levels == 1:
            if interval_min_ms <= 0 <= interval_max_ms:
                for f in levels[0]:
                    results.append(_make_result(video_id, youtube_id, fps, f, float(f.get("score", 0.0))))
            continue

        last_level = num_levels - 1
        mid_levels = list(range(1, last_level))  # empty when num_levels == 2

        for start in levels[0]:
            start_time = start["timestamp_ms"]
            start_score = float(start.get("score", 0.0))

            # Frontier chain state seeded at `start`. Empty path here means
            # "no middle frames yet" — used as-is when num_levels == 2.
            # `frontier_paths` mirrors the other two arrays (same index `p`)
            # so the full chain can be reconstructed for the `steps` field,
            # not just the closing frame's score.
            frontier_scores = [start_score]
            frontier_frames = [start]
            frontier_paths = [[start]]

            for lvl in mid_levels:
                offset = offsets[lvl]
                cur_frames = levels[lvl]

                best_score = float("-inf")
                best_idx = -1
                p = 0
                next_scores = []
                next_frames = []
                next_paths = []
                for f in cur_frames:
                    threshold = f["timestamp_ms"] - offset
                    while p < len(frontier_frames) and frontier_frames[p]["timestamp_ms"] <= threshold:
                        if frontier_scores[p] > best_score:
                            best_score = frontier_scores[p]
                            best_idx = p
                        p += 1
                    if best_idx >= 0:
                        next_scores.append(best_score + float(f.get("score", 0.0)))
                        next_frames.append(f)
                        next_paths.append(frontier_paths[best_idx] + [f])
                frontier_scores = next_scores
                frontier_frames = next_frames
                frontier_paths = next_paths
                if not frontier_frames:
                    break  # no way to continue the chain from this start

            if not frontier_frames:
                continue

            # Close the chain against every candidate end frame. Both
            # `frontier_frames` and `levels[last_level]` are sorted by
            # timestamp_ms, so the valid-predecessor window only grows as
            # `end` advances (same two-pointer trick as the DP above), and
            # once span exceeds interval_max_ms no later `end` can qualify.
            end_offset = offsets[last_level]
            best_score = float("-inf")
            best_idx = -1
            p = 0
            for end in levels[last_level]:
                span_ms = end["timestamp_ms"] - start_time
                if span_ms > interval_max_ms:
                    break

                threshold = end["timestamp_ms"] - end_offset
                while p < len(frontier_frames) and frontier_frames[p]["timestamp_ms"] <= threshold:
                    if frontier_scores[p] > best_score:
                        best_score = frontier_scores[p]
                        best_idx = p
                    p += 1

                if span_ms < interval_min_ms or best_idx < 0:
                    continue

                total_score = best_score + float(end.get("score", 0.0))
                full_path = frontier_paths[best_idx] + [end]
                results.append(_make_result(video_id, youtube_id, fps, end, total_score / num_levels, full_path))

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
