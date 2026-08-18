import numpy as np
import pytest
from numpy_ref import brute_force_search
from vectorforge import FlatIndex

rng = np.random.default_rng(0)


def _assert_match(index: FlatIndex, db: np.ndarray, queries: np.ndarray, k: int) -> None:
    ref_ids, ref_d = brute_force_search(db, queries, k, index.metric)
    ids, dists = index.search(queries, k=k)
    np.testing.assert_array_equal(ids, ref_ids)
    finite = np.isfinite(ref_d)
    np.testing.assert_allclose(dists[finite], ref_d[finite], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.isfinite(dists), finite)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_random_vectors_match_numpy(metric: str) -> None:
    dim = 16
    db = rng.standard_normal((64, dim), dtype=np.float32)
    queries = rng.standard_normal((8, dim), dtype=np.float32)
    index = FlatIndex(dim=dim, metric=metric)
    index.add(db)
    _assert_match(index, db, queries, k=10)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_k_equals_one(metric: str) -> None:
    dim = 8
    db = rng.standard_normal((20, dim), dtype=np.float32)
    queries = rng.standard_normal((5, dim), dtype=np.float32)
    index = FlatIndex(dim=dim, metric=metric)
    index.add(db)
    _assert_match(index, db, queries, k=1)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_k_equals_n(metric: str) -> None:
    dim = 4
    db = rng.standard_normal((7, dim), dtype=np.float32)
    queries = rng.standard_normal((3, dim), dtype=np.float32)
    index = FlatIndex(dim=dim, metric=metric)
    index.add(db)
    _assert_match(index, db, queries, k=db.shape[0])


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_duplicated_vectors_are_deterministic(metric: str) -> None:
    dim = 5
    row = rng.standard_normal((1, dim), dtype=np.float32)
    db = np.repeat(row, 6, axis=0)
    queries = row.copy()
    index = FlatIndex(dim=dim, metric=metric)
    index.add(db)
    ids, dists = index.search(queries, k=6)
    np.testing.assert_array_equal(ids[0], np.arange(6))
    np.testing.assert_allclose(dists[0], 0.0, atol=1e-6)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_zero_vectors(metric: str) -> None:
    dim = 6
    db = np.zeros((4, dim), dtype=np.float32)
    db[1] = 1.0
    queries = np.zeros((2, dim), dtype=np.float32)
    queries[1, 0] = 2.0
    index = FlatIndex(dim=dim, metric=metric)
    index.add(db)
    _assert_match(index, db, queries, k=3)


def test_single_query_1d_shape() -> None:
    dim = 3
    db = np.eye(dim, dtype=np.float32)
    index = FlatIndex(dim=dim, metric="l2")
    index.add(db)
    ids, dists = index.search(db[0], k=2)
    assert ids.shape == (2,)
    assert dists.shape == (2,)
    assert int(ids[0]) == 0


def test_malformed_dimensions_raise() -> None:
    index = FlatIndex(dim=4, metric="l2")
    with pytest.raises(ValueError):
        index.add(np.zeros((3, 5), dtype=np.float32))
    index.add(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        index.search(np.zeros((2, 3), dtype=np.float32), k=1)
    with pytest.raises(ValueError):
        index.search(np.zeros(5, dtype=np.float32), k=1)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_non_finite_vectors_raise(metric: str) -> None:
    index = FlatIndex(dim=2, metric=metric)
    for bad in (
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[0.0, np.inf]], dtype=np.float32),
        np.array([[0.0, -np.inf]], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            index.add(bad)


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_non_finite_queries_raise(metric: str) -> None:
    index = FlatIndex(dim=2, metric=metric)
    index.add(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    for bad in (
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
        np.array([0.0, -np.inf], dtype=np.float32),
        np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            index.search(bad, k=1)


def test_batch_matches_row_wise_search() -> None:
    dim = 12
    db = rng.standard_normal((30, dim), dtype=np.float32)
    queries = rng.standard_normal((6, dim), dtype=np.float32)
    index = FlatIndex(dim=dim, metric="cosine")
    index.add(db)
    batch_ids, batch_d = index.search(queries, k=5)
    for i, q in enumerate(queries):
        ids, dists = index.search(q, k=5)
        np.testing.assert_array_equal(batch_ids[i], ids)
        np.testing.assert_allclose(batch_d[i], dists, rtol=1e-6, atol=1e-6)
