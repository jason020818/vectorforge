"""Correctness evaluator: FlatIndex vs NumPy brute-force.

This is the Phase 1 gate. ANN recall scoring against FlatIndex is `eval/recall.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from vectorforge import FlatIndex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
from numpy_ref import brute_force_search  # noqa: E402


def run(n: int, nq: int, dim: int, k: int, metric: str, seed: int) -> int:
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n, dim), dtype=np.float32)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)
    index = FlatIndex(dim=dim, metric=metric)
    index.add(db)
    ids, dists = index.search(queries, k=k)
    ref_ids, ref_d = brute_force_search(db, queries, k, metric)

    ids_ok = np.array_equal(ids, ref_ids)
    finite = np.isfinite(ref_d)
    dist_ok = np.allclose(dists[finite], ref_d[finite], rtol=1e-5, atol=1e-6)
    print("VectorForge correctness evaluation")
    print(f"dataset: synthetic n={n} nq={nq} dim={dim} metric={metric} k={k} seed={seed}")
    print(f"id match: {'PASS' if ids_ok else 'FAIL'}")
    print(f"distance match: {'PASS' if dist_ok else 'FAIL'}")
    if not ids_ok:
        mismatch = int(np.sum(ids != ref_ids))
        print(f"mismatched ids: {mismatch}")
    return 0 if ids_ok and dist_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--nq", type=int, default=32)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return run(args.n, args.nq, args.dim, args.k, args.metric, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
