from app.strategies.base_strategy import BaseStrategy


class duy_temporal_search(BaseStrategy):
    """
    Multi-query temporal localization via 2D DP.

    Each query group i contributes a set of candidate frames (tagged with
    _query_group_index by DataProvider's SAMPLE-mode per-group vector search).
    We look for, per video, an increasing-time sequence with one frame per
    query group such that consecutive frames satisfy:

        time[level][j] - time[level-1][k] >= temporal_offset_ms[level]

    maximizing the summed match score. Query groups with an empty
    semantic_query are skipped (they contribute no candidates and no
    constraint).
    """

    name = "Temporal Search v1"
    description = (
        "DP-based temporal localization across multiple query groups: finds, "
        "per video, the highest-scoring increasing-time frame sequence that "
        "satisfies each group's temporal_offset_ms gap."
    )
    author = "Team AIC 2026"
    version = "1.0"

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
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

            # dp[level][j] = {"score": best cumulative score, "path": [frame,...]}
            # ending at levels[level][j], or None if no valid chain reaches it.
            dp: list[list[dict | None]] = [None] * num_levels
            dp[0] = [
                {"score": float(f.get("score", 0.0)), "path": [f]}
                for f in levels[0]
            ]

            for lvl in range(1, num_levels):
                offset = offsets[lvl]
                prev_frames = levels[lvl - 1]
                prev_dp = dp[lvl - 1]
                cur_frames = levels[lvl]

                # Both lists are sorted by timestamp_ms, so as `f` advances the
                # valid-predecessor window only grows — two-pointer instead of
                # rescanning prev_frames for every candidate (O(n) not O(n*m)).
                best_score = float("-inf")
                best_entry = None
                p = 0
                cur_dp: list[dict | None] = []
                for f in cur_frames:
                    threshold = f["timestamp_ms"] - offset
                    while p < len(prev_frames) and prev_frames[p]["timestamp_ms"] <= threshold:
                        candidate = prev_dp[p]
                        if candidate is not None and candidate["score"] > best_score:
                            best_score = candidate["score"]
                            best_entry = candidate
                        p += 1
                    if best_entry is None:
                        cur_dp.append(None)  # no valid predecessor yet for this frame
                    else:
                        cur_dp.append({
                            "score": best_entry["score"] + float(f.get("score", 0.0)),
                            "path": best_entry["path"] + [f],
                        })
                dp[lvl] = cur_dp

            # Every completed chain (one per candidate final-level frame) becomes
            # a result row, represented by its last frame.
            video = videos.get(video_id, {})
            fps = float(video.get("fps", 25.0))
            youtube_id = str(video.get("youtube_id") or "")

            for entry in dp[num_levels - 1]:
                if entry is None:
                    continue
                last_frame = entry["path"][-1]
                results.append({
                    "video_id":        video_id,
                    "youtube_id":      youtube_id,
                    "frame_id":        last_frame["frame_id"],
                    "frame_number":    last_frame["frame_number"],
                    "timestamp_ms":    last_frame["timestamp_ms"],
                    "confidence":      round(entry["score"] / num_levels, 4),
                    "frame_image_url": last_frame["image_url"],
                    "fps":             fps,
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
