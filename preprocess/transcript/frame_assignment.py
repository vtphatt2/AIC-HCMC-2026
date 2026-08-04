"""Choose one representative keyframe for each transcript sentence."""
from __future__ import annotations

from preprocess.transcript.sentences import TranscriptSentence


def frame_timestamp_ms(frame_id: str, fps: float) -> tuple[int, int]:
    """Return ``(frame_number, timestamp_ms)`` for a six-digit frame ID."""
    frame_part = str(frame_id).strip()
    if not frame_part.isdigit():
        raise ValueError(f"Invalid frame_id '{frame_id}'; expected a numeric frame filename stem")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    frame_number = int(frame_part)
    return frame_number, int(frame_number / fps * 1000)


def anchor_frame_id(
    frame_ids: list[str],
    fps: float,
    sentence: TranscriptSentence,
    *,
    frame_timestamps: dict[str, int] | None = None,
) -> str | None:
    """Return the keyframe nearest a sentence midpoint, without leaving it.

    ``None`` means the source keyframe set contains no frame during the
    sentence interval. This is preferable to assigning an unrelated frame.
    """
    candidates: list[tuple[str, int]] = []
    for frame_id in frame_ids:
        if frame_timestamps and frame_id in frame_timestamps:
            timestamp_ms = frame_timestamps[frame_id]
        else:
            _, timestamp_ms = frame_timestamp_ms(frame_id, fps)
        if sentence.start_ms <= timestamp_ms < sentence.end_ms:
            candidates.append((frame_id, timestamp_ms))
    if not candidates:
        return None

    midpoint_ms = (sentence.start_ms + sentence.end_ms) / 2
    return min(candidates, key=lambda item: (abs(item[1] - midpoint_ms), item[0]))[0]
