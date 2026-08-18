"""Ground-truth helper: exact search labels from FlatIndex / NumPy.

Used later to score ANN recall. Phase 0/1 only exposes the exact path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from vectorforge import FlatIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vectors", type=Path, required=True, help="float32 .npy of shape (n, dim)"
    )
    parser.add_argument(
        "--queries", type=Path, required=True, help="float32 .npy of shape (nq, dim)"
    )
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    args = parser.parse_args()

    db = np.load(args.vectors)
    queries = np.load(args.queries)
    if db.ndim != 2 or queries.ndim != 2 or db.shape[1] != queries.shape[1]:
        raise SystemExit("vectors and queries must be 2D with the same dim")

    index = FlatIndex(dim=int(db.shape[1]), metric=args.metric)
    index.add(np.ascontiguousarray(db, dtype=np.float32))
    ids, dists = index.search(np.ascontiguousarray(queries, dtype=np.float32), k=args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=ids, distances=dists, metric=args.metric, k=args.k)
    print(f"wrote {args.out} ids={ids.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
