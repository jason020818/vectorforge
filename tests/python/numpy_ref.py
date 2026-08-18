"""NumPy brute-force reference used by correctness tests.

Distance definitions match the C++ scalar kernels:

* L2: sqrt(sum_i (a_i - b_i)^2) accumulated in float64, stored as float32
* cosine distance: 1 - cosine_similarity, with similarity 0 when either vector
  has zero L2 norm
"""

from __future__ import annotations

import numpy as np

IDX_PAD = np.int64(-1)
DIST_PAD = np.float32(np.inf)


def as_f32_2d(x: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"{name} must have shape (n, {dim}), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def l2_distances(db: np.ndarray, queries: np.ndarray) -> np.ndarray:
    db64 = db.astype(np.float64, copy=False)
    q64 = queries.astype(np.float64, copy=False)
    # (nq, 1, d) - (1, n, d) -> (nq, n, d)
    diff = q64[:, None, :] - db64[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1)).astype(np.float32)


def cosine_distances(db: np.ndarray, queries: np.ndarray) -> np.ndarray:
    db64 = db.astype(np.float64, copy=False)
    q64 = queries.astype(np.float64, copy=False)
    dots = q64 @ db64.T
    qn = np.sqrt((q64 * q64).sum(axis=1, keepdims=True))
    dn = np.sqrt((db64 * db64).sum(axis=1, keepdims=True)).T
    denom = qn * dn
    sim = np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0.0)
    return (1.0 - sim).astype(np.float32)


def topk(distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, distances) with shape (nq, k), padded if n < k.

    Ties break toward the smaller id, matching FlatIndex.
    """
    nq, n = distances.shape
    ids = np.empty((nq, k), dtype=np.int64)
    out_d = np.empty((nq, k), dtype=np.float32)
    order_ids = np.arange(n, dtype=np.int64)
    for q in range(nq):
        # lexsort: last key is primary. Sort by distance, then id.
        order = np.lexsort((order_ids, distances[q]))
        take = min(k, n)
        ids[q, :take] = order[:take]
        out_d[q, :take] = distances[q, order[:take]]
        if take < k:
            ids[q, take:] = IDX_PAD
            out_d[q, take:] = DIST_PAD
    return ids, out_d


def brute_force_search(
    db: np.ndarray, queries: np.ndarray, k: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    metric = metric.lower()
    if metric in {"l2", "euclidean"}:
        dists = l2_distances(db, queries)
    elif metric in {"cosine", "cos"}:
        dists = cosine_distances(db, queries)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return topk(dists, k)
