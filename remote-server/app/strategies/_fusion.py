def rrf(rankings, *, key, k=60):
    if k < 1:
        raise ValueError("k must be positive")
    merged, scores, evidence = {}, {}, {}
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
    return sorted(output, key=lambda row: row["rrf_score"], reverse=True)
