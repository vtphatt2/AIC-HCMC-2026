"""Second-phase relevance scoring for Video view's expanded frame strips.

Not a BaseStrategy — it re-scores an already-assembled set of frames
(a search's own matches plus context-frames.py's expanded neighbors)
against the query events that produced that search, rather than
retrieving anything itself. Lives alongside _fusion.py/_duy_temporal_core.py
as shared strategy-level infrastructure precisely so this stays one
consistent mechanism instead of being special-cased into the Video-view
endpoint that calls it.
"""
import numpy as np

from ._duy_temporal_core import resolve_event_weights


def event_weighted_scores(
    query_vectors: list[list[float]],
    frame_vectors: dict[str, list[float]],
    *,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """score(frame) = sum_i weight_i * cosine_sim(frame, query_i).

    `weights` is the same event_weights duy_temporal_search and
    duy_multi_detail_search already use — a single-query search
    degenerates to one term (weight 1), so this formula is identical
    across every strategy, not a Video-view special case.

    The point: a strategy's own result score (RRF confidence, a temporal
    chain's DP score, ...) is on a different scale per strategy, so it
    can't be compared against a context frame that was never part of any
    ranking at all. This gives every frame — original match or expanded
    neighbor — one directly comparable score, computed the same way.
    """
    if not query_vectors or not frame_vectors:
        return {frame_id: 0.0 for frame_id in frame_vectors}

    q = np.asarray(query_vectors, dtype="float32")
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    q = q / q_norms
    w = np.asarray(resolve_event_weights(weights, len(query_vectors)), dtype="float32")

    scores: dict[str, float] = {}
    for frame_id, vector in frame_vectors.items():
        v = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            scores[frame_id] = 0.0
            continue
        sims = q @ (v / norm)
        scores[frame_id] = float(np.dot(w, sims))
    return scores
