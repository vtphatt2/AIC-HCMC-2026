"""Configurable shot-length based keyframe sampling."""
from __future__ import annotations

from bisect import bisect_left
from typing import Any, Callable, Iterable, Mapping, Sequence

from preprocess.keyframes.contracts import (
    FrameCandidate,
    KeyframeSelector,
    SceneSegment,
    SelectedFrame,
    VideoInfo,
)


class LinearRuleBasedSelector(KeyframeSelector):
    """Select uniformly placed representatives using an injected count rule.

    The selector knows how to place frames inside a segment, while the caller
    injects the duration-to-count rule.  This keeps thresholds and increments
    in configuration rather than in the selector implementation.
    """

    name = "linear_rulebase"
    version = "1"

    def __init__(
        self,
        segments: Sequence[SceneSegment],
        *,
        frame_count_for: Callable[[int], int],
        rule_config: Mapping[str, Any],
    ) -> None:
        ordered = sorted(segments, key=lambda segment: (segment.start_ms, segment.end_ms))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ms < previous.end_ms:
                raise ValueError("Scene segments must not overlap")
        self.segments = tuple(ordered)
        self.frame_count_for = frame_count_for
        self.rule_config = dict(rule_config)

    def select(
        self,
        video: VideoInfo,
        candidates: Iterable[FrameCandidate],
    ) -> Sequence[SelectedFrame]:
        del video  # The rule is defined by each detector-provided segment.
        inventory = sorted(
            candidates,
            key=lambda candidate: (
                candidate.ref.timestamp_ms,
                candidate.ref.source_frame_number,
            ),
        )
        timestamps = [candidate.ref.timestamp_ms for candidate in inventory]
        selected: list[SelectedFrame] = []
        used_frame_numbers: set[int] = set()

        for segment_index, segment in enumerate(self.segments):
            duration_ms = segment.end_ms - segment.start_ms
            count = self.frame_count_for(duration_ms)
            if count <= 0:
                raise ValueError(f"Frame-count rule returned a non-positive count: {count}")
            start_index = bisect_left(timestamps, segment.start_ms)
            end_index = bisect_left(timestamps, segment.end_ms)
            if start_index >= end_index:
                continue

            for sample_index in range(count):
                target_ms = segment.start_ms + round((sample_index + 0.5) * duration_ms / count)
                best = min(
                    (
                        candidate
                        for candidate in inventory[start_index:end_index]
                        if candidate.ref.source_frame_number not in used_frame_numbers
                    ),
                    key=lambda candidate: (
                        abs(candidate.ref.timestamp_ms - target_ms),
                        candidate.ref.source_frame_number,
                    ),
                    default=None,
                )
                if best is None:
                    break
                used_frame_numbers.add(best.ref.source_frame_number)
                selected.append(
                    SelectedFrame(
                        ref=best.ref,
                        score=1.0,
                        reasons=("linear_rulebase",),
                        rank=len(selected),
                        metadata={
                            "scene_index": segment_index,
                            "scene_start_ms": segment.start_ms,
                            "scene_end_ms": segment.end_ms,
                            "target_ms": target_ms,
                            "rule": self.rule_config,
                        },
                    )
                )

        return selected
