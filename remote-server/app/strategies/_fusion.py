def rrf(rankings, *, key, k=60, weights=None):
    if k < 1:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    if any(float(weight) < 0 for weight in weights):
        raise ValueError("weights must not be negative")
    merged, scores, evidence = {}, {}, {}
    for ranking, weight in zip(rankings, weights):
        if float(weight) == 0:
            continue
        seen = set()
        for rank, hit in enumerate(ranking, start=1):
            identity = hit.get(key)
            if identity is None or identity in seen:
                continue
            seen.add(identity)
            merged.setdefault(identity, dict(hit))
            scores[identity] = scores.get(identity, 0.0) + float(weight) / (k + rank)
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
    return sorted(output, key=lambda row: row["rrf_score"], reverse=True)
