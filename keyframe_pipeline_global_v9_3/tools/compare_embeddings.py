#!/usr/bin/env python3
"""Compare two pipeline output directories (baseline vs optimized embeddings)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path, help="FP32/strict baseline output directory")
    ap.add_argument("candidate", type=Path, help="AMP/TF32 candidate output directory")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    rows=[]
    for bp in sorted(args.baseline.glob("*/embeddings.npy")):
        rel=bp.relative_to(args.baseline)
        cp=args.candidate/rel
        if not cp.exists():
            continue
        a=np.load(bp).astype(np.float32, copy=False)
        b=np.load(cp).astype(np.float32, copy=False)
        if a.shape != b.shape:
            print(f"skip shape mismatch {rel}: {a.shape} vs {b.shape}")
            continue
        # persisted vectors should already be normalized; renormalize defensively.
        an=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        bn=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
        cos=np.sum(an*bn,axis=1)
        rows.append((str(rel.parent), len(cos), float(cos.mean()), float(cos.min()), float(np.percentile(cos,1))))

    if not rows:
        raise SystemExit("No matching embeddings.npy files found")
    n=sum(r[1] for r in rows)
    weighted=sum(r[1]*r[2] for r in rows)/n
    result={
        "matched_videos":len(rows), "matched_embeddings":n,
        "weighted_mean_cosine":weighted,
        "minimum_cosine":min(r[3] for r in rows),
        "p1_min_across_videos":min(r[4] for r in rows),
        "videos":[{"video_id":r[0],"n":r[1],"mean_cosine":r[2],"min_cosine":r[3],"p1_cosine":r[4]} for r in rows],
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
