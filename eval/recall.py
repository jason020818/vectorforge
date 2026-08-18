"""ANN recall evaluator: HNSWIndex vs FlatIndex ground-truth IDs.

Recall@K = mean over queries of |HNSW top-K IDs ∩ FlatIndex top-K IDs| / K.

This is the Phase 2 correctness gate. Distances are not compared approximately;
ground truth is the exact FlatIndex ID set.
"""

from __future__ import annotations

import argparse

import numpy as np
from vectorforge import FlatIndex, HNSWIndex

# Documented Phase 2 synthetic validation set.
DEFAULT_N = 10000
DEFAULT_DIM = 64
DEFAULT_NQ = 100
DEFAULT_SEED = 42
DEFAULT_M = 16
DEFAULT_EFC = 200
DEFAULT_EFS = 100
RECALL10_GATE = 0.90


def recall_at_k(ann_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    nq = ann_ids.shape[0]
    hits = 0
    for i in range(nq):
        truth = set(int(x) for x in gt_ids[i, :k].tolist())
        hits += sum(1 for x in ann_ids[i, :k].tolist() if int(x) in truth)
    return hits / float(nq * k)


def run(
    n: int,
    nq: int,
    dim: int,
    seed: int,
    M: int,
    ef_construction: int,
    ef_search: int,
    metric: str,
) -> int:
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n, dim), dtype=np.float32)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)

    flat = FlatIndex(dim=dim, metric=metric)
    hnsw = HNSWIndex(
        dim=dim,
        metric=metric,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        seed=seed,
    )
    flat.add(db)
    hnsw.add(db)

    gt10, _ = flat.search(queries, k=10)
    ann10, _ = hnsw.search(queries, k=10)
    gt100, _ = flat.search(queries, k=100)
    ann100, _ = hnsw.search(queries, k=100)
    r10 = recall_at_k(ann10, gt10, 10)
    r100 = recall_at_k(ann100, gt100, 100)

    print("VectorForge HNSW recall evaluation")
    print(
        f"dataset: synthetic n={n} nq={nq} dim={dim} metric={metric} "
        f"seed={seed} M={M} efConstruction={ef_construction} efSearch={ef_search}"
    )
    print(f"Recall@10: {r10:.6f}")
    print(f"Recall@100: {r100:.6f}")
    print(f"Recall@10 gate ({RECALL10_GATE:.2f}): {'PASS' if r10 >= RECALL10_GATE else 'FAIL'}")
    return 0 if r10 >= RECALL10_GATE else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--nq", type=int, default=DEFAULT_NQ)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--ef-construction", type=int, default=DEFAULT_EFC)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EFS)
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    args = parser.parse_args()
    return run(
        args.n,
        args.nq,
        args.dim,
        args.seed,
        args.M,
        args.ef_construction,
        args.ef_search,
        args.metric,
    )


if __name__ == "__main__":
    raise SystemExit(main())
