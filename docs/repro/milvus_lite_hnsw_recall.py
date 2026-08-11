#!/usr/bin/env python3
"""Standalone reproduction: milvus-lite HNSW returns wrong nearest neighbours.

Self-contained — synthetic vectors, a temporary database, no external data.
Run it, and it prints the evidence needed for an upstream bug report.

    pip install "pymilvus" "milvus-lite" "faiss-cpu" numpy
    python milvus_lite_hnsw_recall.py

What it shows:

  1. HNSW recall@10 against exact brute force, for queries drawn from the same
     distribution as the data AND for queries drawn from a different one.
  2. `ef` sweeps that change nothing, because the search executor never
     forwards search params to the index.
  3. The two lines responsible, and that patching both restores correct results.

Root cause:

  milvus_lite/search/executor_indexed.py
      local_ids, dists = index.search(query_vectors, top_k, valid_mask=local_mask)
                                                            # params= is never passed

  milvus_lite/index/faiss_hnsw.py
      params = params or {}
      ef = int(params.get("ef", 64))     # therefore always 64
      self._index.hnsw.efSearch = ef

  The gRPC servicer parses `search_params` correctly and FaissHnswIndex.search
  accepts them; only the call between the two drops them. On top of that the
  executor always passes `valid_mask`, routing faiss through its IDSelector
  path, which bounds traversal so that even a forced `ef` has no effect.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

N_VECTORS = 50_000
DIM = 128
TOP_K = 10
SEED = 0

rng = np.random.default_rng(SEED)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def make_data():
    """Vectors clustered on part of the sphere, so an off-cluster query sits
    outside the data the way a CLIP text embedding sits outside image ones."""
    centre = unit(rng.normal(size=DIM))
    data = unit(centre + 0.35 * rng.normal(size=(N_VECTORS, DIM)))
    in_dist = unit(centre + 0.35 * rng.normal(size=DIM))       # same cluster
    out_dist = unit(rng.normal(size=DIM))                       # elsewhere
    return data.astype("float32"), in_dist.astype("float32"), out_dist.astype("float32")


def brute_force(data, q, k):
    scores = data @ q
    idx = np.argsort(-scores)[:k]
    return list(zip(idx.tolist(), scores[idx].tolist()))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="milvus_lite_repro_"))
    uri = str(tmp / "repro.db")
    data, q_in, q_out = make_data()

    client = MilvusClient(uri=uri)
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
    ])
    client.create_collection("repro", schema=schema)
    client.insert("repro", [{"id": i, "vector": data[i].tolist()} for i in range(N_VECTORS)])

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="HNSW",
                           metric_type="COSINE", params={"M": 16, "efConstruction": 256})
    client.create_index("repro", index_params)
    client.load_collection("repro")

    import pymilvus, milvus_lite
    print(f"pymilvus {pymilvus.__version__}   milvus_lite {milvus_lite.__version__}")
    print(f"{N_VECTORS} vectors, dim {DIM}, HNSW M=16 efConstruction=256, COSINE\n")

    def hnsw(q, ef, limit=TOP_K):
        res = client.search("repro", data=[q.tolist()], limit=limit,
                            search_params={"metric_type": "COSINE", "params": {"ef": ef}},
                            output_fields=["id"])[0]
        return [(r["id"], r["distance"]) for r in res]

    for label, q in (("query from the data's own cluster", q_in),
                     ("query from elsewhere on the sphere", q_out)):
        truth = brute_force(data, q, TOP_K)
        truth_ids = {i for i, _ in truth}
        print(f"--- {label} ---")
        print(f"  exact best score: {truth[0][1]:.4f}")
        print(f"  {'ef':>7} {'hnsw best':>10} {'recall@10':>10} {'rows':>6}")
        for ef in (16, 64, 256, 1024, 4096, 16384):
            got = hnsw(q, ef)
            recall = len(truth_ids & {i for i, _ in got}) / TOP_K
            print(f"  {ef:>7} {got[0][1]:>10.4f} {recall:>10.0%} {len(got):>6}")
        deep = hnsw(q, 16384, limit=N_VECTORS)
        print(f"  reachable at limit={N_VECTORS}: {len(deep)} of {N_VECTORS} "
              f"({len(deep)/N_VECTORS:.1%})\n")

    # ---- confirm the cause by forcing both dropped inputs -------------------
    from milvus_lite.index.faiss_hnsw import FaissHnswIndex
    original = FaissHnswIndex.search
    cfg = {"ef": 4096, "drop_mask": True}

    def patched(self, queries, top_k, valid_mask=None, params=None):
        return original(self, queries, top_k,
                        valid_mask=None if cfg["drop_mask"] else valid_mask,
                        params={"ef": cfg["ef"]})

    FaissHnswIndex.search = patched
    print("--- with params forwarded AND valid_mask dropped (patched) ---")
    for label, q in (("from cluster", q_in), ("from elsewhere", q_out)):
        truth = brute_force(data, q, TOP_K)
        truth_ids = {i for i, _ in truth}
        got = hnsw(q, 4096)
        recall = len(truth_ids & {i for i, _ in got}) / TOP_K
        print(f"  {label:<16} exact {truth[0][1]:.4f}  hnsw {got[0][1]:.4f}  "
              f"recall@10 {recall:.0%}")
    FaissHnswIndex.search = original

    client.close()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
