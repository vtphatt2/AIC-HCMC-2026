"""Greedy, sequence-aware duplicate filtering for ranked strategy results."""
from __future__ import annotations

import numpy as np


def result_frame_ids(result: dict) -> list[str]:
    steps = result.get("steps")
    rows = steps if isinstance(steps, list) and steps else [result]
    return [str(row["frame_id"]) for row in rows if row.get("frame_id")]


def filter_similar_results(
    results: list[dict],
    embeddings: dict[str, list[float]],
    *,
    threshold: float,
    event_weights: list[float] | None = None,
) -> list[dict]:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("duplicate_threshold must be between 0 and 1")

    normalized = {}
    for frame_id, value in embeddings.items():
        vector = np.asarray(value, dtype="float32").reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm:
            normalized[str(frame_id)] = vector / norm

    kept = []
    kept_vectors = []
    for result in results:
        vectors = [normalized.get(frame_id) for frame_id in result_frame_ids(result)]
        weights = [
            float(event_weights[index]) if event_weights and index < len(event_weights) else 1.0
            for index in range(len(vectors))
        ]
        duplicate = any(
            len(previous) == len(vectors)
            and all(
                left is not None
                and right is not None
                and 1.0 - weight * (1.0 - float(np.dot(left, right))) > threshold + 1e-6
                for left, right, weight in zip(previous, vectors, weights)
            )
            for previous in kept_vectors
        )
        if not duplicate:
            kept.append(result)
            kept_vectors.append(vectors)
    return kept
