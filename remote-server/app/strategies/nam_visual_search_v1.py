from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class NamVisualSearchV1(BaseStrategy):
    name = "Nam Visual Search v1"
    description = "PE-Core visual similarity ranking with temporal multi-step fusion."
    author = "Nam"
    version = "1.0"

    TEMPORAL_TOLERANCE_MS = 3000
    PER_STEP_CANDIDATES = 300

    def pre_process(self, query_groups: list[dict]) -> list[dict]:
        processed = []
        for index, group in enumerate(query_groups):
            semantic_query = str(group.get("semantic_query", "")).strip()
            text_query = str(group.get("text_query", "")).strip()
            temporal_offset_ms = int(group.get("temporal_offset_ms") or 0)
            if index == 0:
                temporal_offset_ms = 0

            processed.append(
                {
                    **group,
                    "semantic_query": semantic_query,
                    "text_query": text_query,
                    "temporal_offset_ms": temporal_offset_ms,
                }
            )

        if not any(group["semantic_query"] for group in processed):
            raise ValueError("Nam Visual Search v1 requires at least one semantic_query.")
        return processed

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames = raw_data.get("frames", [])
        videos = raw_data.get("videos", {})
        if not frames:
            logger.warning("NamVisualSearchV1 received no frames from DataProvider")
            return []

        candidate_limit = self.PER_STEP_CANDIDATES if len(query_groups) > 1 else len(frames)
        grouped_candidates = self._group_candidates_by_query_step(
            frames,
            query_groups,
            videos,
            candidate_limit,
        )
        if len(query_groups) == 1:
            return [candidate["result"] for candidate in grouped_candidates.get(0, [])]

        return self._temporal_fusion(grouped_candidates, query_groups)

    def _group_candidates_by_query_step(
        self,
        frames: list[dict[str, Any]],
        query_groups: list[dict],
        videos: dict[str, dict],
        candidate_limit: int,
    ) -> dict[int, list[dict[str, Any]]]:
        has_group_tags = any("_query_group_index" in frame for frame in frames)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for frame in frames:
            score = self._visual_score(frame)
            result = self._format_result(frame, videos, score)
            group_index = int(frame.get("_query_group_index", 0)) if has_group_tags else 0
            grouped[group_index].append(
                {
                    "frame": frame,
                    "score": score,
                    "result": result,
                }
            )

        if not has_group_tags and len(query_groups) > 1:
            grouped[0].sort(key=lambda item: item["score"], reverse=True)
            for group_index in range(1, len(query_groups)):
                grouped[group_index] = list(grouped[0])

        for group_index, candidates in grouped.items():
            candidates.sort(key=lambda item: item["score"], reverse=True)
            grouped[group_index] = candidates[:candidate_limit]
        return grouped

    def _temporal_fusion(
        self,
        grouped_candidates: dict[int, list[dict[str, Any]]],
        query_groups: list[dict],
    ) -> list[dict]:
        anchors = grouped_candidates.get(0, [])
        if not anchors:
            return []

        cumulative_offsets = self._cumulative_offsets(query_groups)
        candidates_by_group_video = {
            group_index: self._index_by_video(candidates)
            for group_index, candidates in grouped_candidates.items()
        }

        fused_results = []
        for anchor in anchors:
            anchor_result = anchor["result"]
            combined_score = anchor["score"]
            matched_steps = 1

            for group_index in range(1, len(query_groups)):
                expected_ts = int(anchor_result["timestamp_ms"]) + cumulative_offsets[group_index]
                video_candidates = candidates_by_group_video.get(group_index, {}).get(anchor_result["video_id"], [])
                match = self._best_temporal_match(video_candidates, expected_ts)
                if match is None:
                    combined_score *= 0.65
                    continue

                gap = abs(int(match["result"]["timestamp_ms"]) - expected_ts)
                temporal_score = max(0.0, 1.0 - (gap / self.TEMPORAL_TOLERANCE_MS))
                combined_score += 0.85 * match["score"] + 0.15 * temporal_score
                matched_steps += 1

            confidence = combined_score / max(1, matched_steps)
            fused = dict(anchor_result)
            fused["confidence"] = round(self._clamp01(confidence), 4)
            fused_results.append(fused)

        fused_results.sort(key=lambda row: row["confidence"], reverse=True)
        return fused_results

    @staticmethod
    def _cumulative_offsets(query_groups: list[dict]) -> list[int]:
        offsets = [0]
        running = 0
        for group in query_groups[1:]:
            running += int(group.get("temporal_offset_ms") or 0)
            offsets.append(running)
        return offsets

    @staticmethod
    def _index_by_video(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_video[candidate["result"]["video_id"]].append(candidate)
        for rows in by_video.values():
            rows.sort(key=lambda item: item["result"]["timestamp_ms"])
        return by_video

    def _best_temporal_match(
        self,
        candidates: list[dict[str, Any]],
        expected_timestamp_ms: int,
    ) -> dict[str, Any] | None:
        best = None
        best_score = -1.0
        for candidate in candidates:
            gap = abs(int(candidate["result"]["timestamp_ms"]) - expected_timestamp_ms)
            if gap > self.TEMPORAL_TOLERANCE_MS:
                continue
            temporal_score = 1.0 - (gap / self.TEMPORAL_TOLERANCE_MS)
            score = 0.85 * candidate["score"] + 0.15 * temporal_score
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _format_result(self, frame: dict[str, Any], videos: dict[str, dict], score: float) -> dict[str, Any]:
        video = videos.get(frame.get("video_id"), {})
        return {
            "video_id": str(frame.get("video_id", "")),
            "youtube_id": str(video.get("youtube_id") or ""),
            "frame_id": str(frame.get("frame_id", "")),
            "frame_number": int(frame.get("frame_number") or 0),
            "timestamp_ms": int(frame.get("timestamp_ms") or 0),
            "confidence": round(self._clamp01(score), 4),
            "frame_image_url": str(frame.get("image_url") or frame.get("frame_image_url") or ""),
            "fps": float(video.get("fps", 25.0)),
        }

    def _visual_score(self, frame: dict[str, Any]) -> float:
        raw_score = frame.get("score", frame.get("confidence", 0.5))
        try:
            return self._clamp01(float(raw_score))
        except (TypeError, ValueError):
            logger.exception("Invalid visual score on frame_id=%s", frame.get("frame_id"))
            return 0.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))
