"""TransNetV2 scene-segment sampling described in ``docs/keyframe_selection.md``."""
from __future__ import annotations

from bisect import bisect_left
from typing import Iterable, Sequence

from preprocess.keyframes.contracts import (
    FrameCandidate,
    KeyframeSelector,
    SceneSegment,
    SelectedFrame,
    VideoInfo,
)


class SceneSegmentSelector(KeyframeSelector):
    """Sample 1, 3, or 5 representatives at proportional scene positions.

    The policy mirrors the project documentation: scenes up to 3 seconds get
    one sample, scenes up to 10 seconds get three, and longer scenes get five.
    Sample positions are centred inside equal-duration buckets, e.g. 10/30/50/
    70/90 percent for a five-sample scene.
    """

    name = "transnetv2_scene_uniform"
    version = "1"

    def __init__(self, segments: Sequence[SceneSegment]) -> None:
        ordered = sorted(segments, key=lambda segment: (segment.start_ms, segment.end_ms))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ms < previous.end_ms:
                raise ValueError("Scene segments must not overlap")
        self.segments = tuple(ordered)

    @staticmethod
    def _sample_count(duration_ms: int) -> int:
        if duration_ms <= 3_000:
            return 1
        if duration_ms <= 10_000:
            return 3
        return 5

    def select(
        self,
        video: VideoInfo,
        candidates: Iterable[FrameCandidate],
    ) -> Sequence[SelectedFrame]:
        # The selector only retains lightweight frame references; no decoded
        # pixel data is required for this documented rule-based strategy.
        inventory = list(candidates)
        timestamps = [candidate.ref.timestamp_ms for candidate in inventory]
        selected: list[SelectedFrame] = []
        used_frame_numbers: set[int] = set()

        for segment_index, segment in enumerate(self.segments):
            duration_ms = segment.end_ms - segment.start_ms
            count = self._sample_count(duration_ms)
            for sample_index in range(count):
                target_ms = segment.start_ms + round((sample_index + 0.5) * duration_ms / count)
                insertion_point = bisect_left(timestamps, target_ms)
                candidate_indexes = (insertion_point - 1, insertion_point)
                valid_indexes = [
                    index for index in candidate_indexes
                    if 0 <= index < len(inventory)
                    and segment.start_ms <= inventory[index].ref.timestamp_ms < segment.end_ms
                    and inventory[index].ref.source_frame_number not in used_frame_numbers
                ]
                if not valid_indexes:
                    # A detector segment can contain no decoded presentation
                    # frame (for example after timestamp rounding).  Skipping is
                    # safer than selecting a frame from a neighbouring scene.
                    continue
                best_index = min(
                    valid_indexes,
                    key=lambda index: (
                        abs(inventory[index].ref.timestamp_ms - target_ms),
                        inventory[index].ref.source_frame_number,
                    ),
                )
                candidate = inventory[best_index]
                used_frame_numbers.add(candidate.ref.source_frame_number)
                selected.append(
                    SelectedFrame(
                        ref=candidate.ref,
                        score=1.0,
                        reasons=("scene_segment_uniform",),
                        rank=len(selected),
                        metadata={
                            "scene_index": segment_index,
                            "scene_start_ms": segment.start_ms,
                            "scene_end_ms": segment.end_ms,
                            "target_ms": target_ms,
                        },
                    )
                )
        return selected
