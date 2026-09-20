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

    frame_ids = [result_frame_ids(result) for result in results]
    if results and all(len(ids) == 1 for ids in frame_ids):
        return _filter_single_frames(results, frame_ids, normalized, float(threshold))

    kept = []
    kept_vectors = []
    for result in results:
        vectors = [normalized.get(frame_id) for frame_id in result_frame_ids(result)]
        # event_weights means "how important is step i of a multi-step chain"
        # (temporal/TRAKE) — meaningful only when a result actually has more
        # than one frame. A single-frame result (raw_visual, or
        # duy_multi_detail_search's RRF-fused single frame) has no "step"
        # for event_weights[0] to legitimately refer to; treating it as one
        # anyway means a strategy's per-detail RANKING weight (RRF) silently
        # doubles as dedup strictness for every result, which is a bug, not
        # a feature — hence the len(vectors) > 1 gate.
        weights = (
            [
                float(event_weights[index]) if event_weights and index < len(event_weights) else 1.0
                for index in range(len(vectors))
            ]
            if len(vectors) > 1
            else [1.0] * len(vectors)
        )
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


def _filter_single_frames(results, frame_ids, normalized, threshold):
    """Use a BLAS prefilter, preserving the scalar rule near its boundary.

    Most candidate pairs are clearly different. A matrix multiply discards
    those pairs together; possible duplicates still use the original np.dot
    comparison so rounding at the threshold cannot change the decision.
    Missing/zero vectors are retained, just as in the temporal path.
    """
    valid = [i for i, ids in enumerate(frame_ids) if ids[0] in normalized]
    if not valid:
        return list(results)
    vectors = np.stack([normalized[frame_ids[i][0]] for i in valid])
    similarities = vectors @ vectors.T
    # Conservative float32 dot-product roundoff bound for normalized vectors.
    guard = 4 * np.finfo(np.float32).eps * vectors.shape[1]
    positions = {result_index: vector_index for vector_index, result_index in enumerate(valid)}
    kept, kept_positions = [], []
    cutoff = threshold + 1e-6
    for i, result in enumerate(results):
        position = positions.get(i)
        duplicate = False
        if position is not None and kept_positions:
            candidates = np.asarray(kept_positions)
            candidates = candidates[similarities[position, candidates] > cutoff - guard]
            duplicate = any(
                1.0 - (1.0 - float(np.dot(vectors[position], vectors[previous]))) > cutoff
                for previous in candidates
            )
        if not duplicate:
            kept.append(result)
            if position is not None:
                kept_positions.append(position)
    return kept
