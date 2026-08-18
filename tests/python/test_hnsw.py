from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from vectorforge import FlatIndex, HNSWIndex

rng = np.random.default_rng(0)


def _recall_at_k(ann_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    nq = ann_ids.shape[0]
    hits = 0
    for i in range(nq):
        truth = set(gt_ids[i, :k].tolist())
        hits += sum(1 for x in ann_ids[i, :k].tolist() if x in truth)
    return hits / (nq * k)


def test_construction_and_params() -> None:
    index = HNSWIndex(dim=8, metric="cosine", M=8, ef_construction=32, ef_search=16, seed=7)
    assert index.dim == 8
    assert index.metric == "cosine"
    assert index.M == 8
    assert index.ef_construction == 32
    assert index.ef_search == 16
    assert index.seed == 7
    index.ef_search = 24
    assert index.ef_search == 24
    with pytest.raises(ValueError):
        index.ef_search = 0


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_add_and_search(metric: str) -> None:
    dim = 8
    db = rng.standard_normal((32, dim), dtype=np.float32)
    queries = rng.standard_normal((4, dim), dtype=np.float32)
    index = HNSWIndex(dim=dim, metric=metric, M=8, ef_construction=32, ef_search=16, seed=42)
    index.add(db)
    ids, dists = index.search(queries, k=5)
    assert ids.shape == (4, 5)
    assert dists.shape == (4, 5)
    assert np.all(ids[:, 0] >= 0)


def test_single_query_1d_shape() -> None:
    dim = 4
    db = np.eye(dim, dtype=np.float32)
    index = HNSWIndex(dim=dim, metric="l2", M=4, ef_construction=8, ef_search=8, seed=1)
    index.add(db)
    ids, dists = index.search(db[0], k=2)
    assert ids.shape == (2,)
    assert dists.shape == (2,)
    assert int(ids[0]) == 0


def test_batch_matches_row_wise_search() -> None:
    dim = 6
    db = rng.standard_normal((20, dim), dtype=np.float32)
    queries = rng.standard_normal((5, dim), dtype=np.float32)
    index = HNSWIndex(dim=dim, metric="l2", M=8, ef_construction=32, ef_search=16, seed=3)
    index.add(db)
    batch_ids, batch_d = index.search(queries, k=4)
    for i, q in enumerate(queries):
        ids, dists = index.search(q, k=4)
        np.testing.assert_array_equal(batch_ids[i], ids)
        np.testing.assert_allclose(batch_d[i], dists, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_non_finite_input_raises(metric: str) -> None:
    index = HNSWIndex(dim=2, metric=metric, M=4, ef_construction=8, ef_search=4, seed=1)
    for bad in (
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[0.0, np.inf]], dtype=np.float32),
        np.array([[0.0, -np.inf]], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            index.add(bad)
    index.add(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    for bad in (
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            index.search(bad, k=1)


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        HNSWIndex(dim=0, metric="l2")
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, metric="l2", M=1)
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, metric="l2", M=16, ef_construction=8)
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, metric="l2", ef_search=0)


def test_save_load(tmp_path: Path) -> None:
    path = tmp_path / "hnsw.bin"
    dim = 4
    db = rng.standard_normal((16, dim), dtype=np.float32)
    q = rng.standard_normal((3, dim), dtype=np.float32)
    index = HNSWIndex(dim=dim, metric="cosine", M=6, ef_construction=24, ef_search=12, seed=42)
    index.add(db)
    index.save(os.fspath(path))
    loaded = HNSWIndex(dim=8, metric="l2")
    loaded.load(os.fspath(path))
    assert loaded.dim == dim
    assert loaded.metric == "cosine"
    assert loaded.ntotal == 16
    assert loaded.graph_digest() == index.graph_digest()
    ids_a, dist_a = index.search(q, k=4)
    ids_b, dist_b = loaded.search(q, k=4)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_allclose(dist_a, dist_b, atol=0.0)


def test_recall_against_flat() -> None:
    dim = 16
    n = 300
    nq = 20
    db = rng.standard_normal((n, dim), dtype=np.float32)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)
    flat = FlatIndex(dim=dim, metric="l2")
    hnsw = HNSWIndex(dim=dim, metric="l2", M=12, ef_construction=64, ef_search=32, seed=42)
    flat.add(db)
    hnsw.add(db)
    gt, _ = flat.search(queries, k=10)
    ann, _ = hnsw.search(queries, k=10)
    assert _recall_at_k(ann, gt, 10) >= 0.90
