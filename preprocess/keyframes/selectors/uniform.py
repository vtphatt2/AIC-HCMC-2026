"""Deterministic time-uniform keyframe selection."""
from __future__ import annotations

from typing import Iterable, Sequence

from preprocess.keyframes.contracts import FrameCandidate, KeyframeSelector, SelectedFrame, VideoInfo


class UniformIntervalSelector(KeyframeSelector):
    """Choose the first decoded frame at or after each fixed time boundary.

    A timestamp-based rule is intentionally used instead of ``every N frames``:
    it behaves correctly for videos with different frame rates and for VFR media.
    """

    name = "uniform_interval"
    version = "1"

    def __init__(self, interval_ms: int = 1_000) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.interval_ms = interval_ms

    def select(
        self,
        video: VideoInfo,
        candidates: Iterable[FrameCandidate],
    ) -> Sequence[SelectedFrame]:
        selected: list[SelectedFrame] = []
        next_boundary_ms = 0

        for candidate in candidates:
            timestamp_ms = candidate.ref.timestamp_ms
            if timestamp_ms < next_boundary_ms:
                continue

            selected.append(
                SelectedFrame(
                    ref=candidate.ref,
                    score=1.0,
                    reasons=("uniform_interval",),
                    rank=len(selected),
                    metadata={"interval_ms": self.interval_ms},
                )
            )
            # Advance from the scheduled boundary rather than the selected
            # frame timestamp.  A delayed/VFR frame cannot shift the cadence.
            next_boundary_ms += self.interval_ms
            while next_boundary_ms <= timestamp_ms:
                next_boundary_ms += self.interval_ms

        return selected
