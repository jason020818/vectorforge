"""Minimal FlatIndex micro-benchmark. Not a published performance claim.

Times a single-process exact search so contributors can sanity-check a build.
Median of several runs is printed. Do not copy these numbers into README.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from vectorforge import FlatIndex


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--nq", type=int, default=50)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    db = rng.standard_normal((args.n, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.nq, args.dim), dtype=np.float32)

    t0 = time.perf_counter()
    index = FlatIndex(dim=args.dim, metric=args.metric)
    index.add(db)
    build_s = time.perf_counter() - t0

    for _ in range(args.warmup):
        index.search(queries, k=args.k)

    samples: list[float] = []
    for _ in range(args.repeat):
        t1 = time.perf_counter()
        index.search(queries, k=args.k)
        samples.append(time.perf_counter() - t1)

    elapsed = median(samples)
    qps = args.nq / elapsed if elapsed > 0 else float("inf")
    print("VectorForge FlatIndex micro-benchmark (not a published result)")
    print(f"n={args.n} nq={args.nq} dim={args.dim} k={args.k} metric={args.metric}")
    print(f"build_s={build_s:.6f}")
    print(f"search_s_median={elapsed:.6f}")
    print(f"qps_median={qps:.2f}")
    print(f"repeats={args.repeat} warmup={args.warmup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
