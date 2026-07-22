def rrf(rankings: list[list[dict]], *, key: str, k: int = 60) -> list[dict]:
    """Reciprocal-rank fusion for rankings whose raw score scales may differ."""
    if k < 1:
        raise ValueError("k must be positive")

    merged: dict[object, dict] = {}
    scores: dict[object, float] = {}
    evidence: dict[object, list[dict]] = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, start=1):
            identity = hit.get(key)
            if identity is None or identity in seen:
                continue
            seen.add(identity)
            merged.setdefault(identity, dict(hit))
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            evidence.setdefault(identity, []).append(dict(hit))

    if not scores:
        return []
    best = max(scores.values())
    output = []
    for identity, hit in merged.items():
        hit["rrf_score"] = scores[identity]
        hit["confidence"] = scores[identity] / best
        hit["evidence"] = evidence[identity]
        output.append(hit)
    output.sort(key=lambda row: row["rrf_score"], reverse=True)
    return output
