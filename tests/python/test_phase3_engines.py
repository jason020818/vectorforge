from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from benchmarks.engines import BenchmarkAdapterConfig, create_engine
from benchmarks.engines.faiss import FaissHnswAdapter
from benchmarks.engines.hnswlib import HnswlibAdapter
from benchmarks.engines.usearch import USearchAdapter
from benchmarks.engines.vectorforge import VectorForgeAdapter


def _available(name: str) -> bool:
    if name == "faiss":
        return importlib.util.find_spec("faiss") is not None
    if name == "hnswlib":
        return importlib.util.find_spec("hnswlib") is not None
    if name == "usearch":
        return importlib.util.find_spec("usearch.index") is not None
    return True


@pytest.mark.skipif(
    not (_available("faiss") and _available("hnswlib") and _available("usearch")),
    reason="optional benchmark dependencies not installed",
)
def test_same_query_workload_and_high_overlap_across_engines() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((32, 8), dtype=np.float32)
    queries = rng.standard_normal((4, 8), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    config = BenchmarkAdapterConfig(
        dim=8,
        metric="cosine",
        M=16,
        ef_construction=200,
        ef_search=100,
        threads=1,
    )
    results = {}
    for name in ("vectorforge", "faiss", "hnswlib", "usearch"):
        adapter = create_engine(name)
        adapter.build(vectors, config)
        ids, _ = adapter.search(queries, 10)
        results[name] = np.asarray(ids)

    base = results["vectorforge"]
    for name, ids in results.items():
        assert ids.shape == base.shape
        overlap = (ids == base).sum() / float(base.size)
        assert overlap >= 0.95, f"{name} overlap too low: {overlap:.3f}"


def test_vectorforge_reports_actual_parameters() -> None:
    adapter = VectorForgeAdapter()
    vectors = np.zeros((4, 8), dtype=np.float32)
    config = BenchmarkAdapterConfig(8, "cosine", 16, 200, 100, 1)
    adapter.build(vectors, config)
    params = adapter.actual_parameters()
    assert params["M0"] == 32
    assert params["seed"] == 42
    assert params["dtype"] == "f32"


def test_hnswlib_requests_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeIndex:
        def init_index(self, **kwargs):
            calls["init_index"] = kwargs

        def set_num_threads(self, value):
            calls["set_num_threads"] = value

        def set_ef(self, value):
            calls["set_ef"] = value

        def add_items(self, data, ids, num_threads=-1, replace_deleted=False):
            calls["add_items"] = {"num_threads": num_threads, "shape": data.shape}

        def knn_query(self, data, k=1, num_threads=-1, filter=None):
            calls["knn_query"] = {"k": k, "num_threads": num_threads, "shape": data.shape}
            return (
                np.zeros((data.shape[0], k), dtype=np.int64),
                np.zeros((data.shape[0], k), dtype=np.float32),
            )

    class FakeModule:
        def Index(self, **kwargs):
            calls["Index"] = kwargs
            return FakeIndex()

    adapter = HnswlibAdapter()
    monkeypatch.setattr(adapter, "_require", lambda: FakeModule())
    config = BenchmarkAdapterConfig(8, "cosine", 16, 200, 100, 1)
    adapter.build(np.zeros((4, 8), dtype=np.float32), config)
    adapter.search(np.zeros((2, 8), dtype=np.float32), 10)
    assert calls["set_num_threads"] == 1
    assert calls["add_items"]["num_threads"] == 1
    assert calls["knn_query"]["num_threads"] == 1
    assert adapter.actual_parameters()["num_threads"] == 1


def test_usearch_uses_explicit_f32_and_reports_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeMatches:
        def __init__(self):
            self.keys = np.zeros((2, 10), dtype=np.uint64)
            self.distances = np.zeros((2, 10), dtype=np.float32)

    class FakeIndex:
        def __init__(self, **kwargs):
            calls["Index"] = kwargs
            self.expansion_search = kwargs["expansion_search"]

        def add(self, keys, vectors, *, copy=True, threads=0, log=False, progress=None, dtype=None):
            calls["add"] = {"threads": threads, "dtype": dtype, "shape": vectors.shape}

        def search(
            self,
            vectors,
            count=10,
            *,
            threads=0,
            exact=False,
            log=False,
            progress=None,
            dtype=None,
        ):
            calls["search"] = {
                "threads": threads,
                "dtype": dtype,
                "count": count,
                "shape": vectors.shape,
            }
            return FakeMatches()

        def save(self, path):
            calls["save"] = path

    class FakeModule:
        Index = FakeIndex

    adapter = USearchAdapter()
    monkeypatch.setattr(adapter, "_require", lambda: FakeModule())
    config = BenchmarkAdapterConfig(8, "cosine", 16, 200, 100, 1)
    adapter.build(np.zeros((4, 8), dtype=np.float32), config)
    adapter.search(np.zeros((2, 8), dtype=np.float32), 10)
    assert calls["Index"]["dtype"] == "f32"
    assert calls["add"]["threads"] == 1
    assert calls["search"]["threads"] == 1
    params = adapter.actual_parameters()
    assert params["dtype"] == "f32"
    assert params["thread_control_status"] == "explicit_threads_parameter"


def test_faiss_reports_actual_parameters() -> None:
    adapter = FaissHnswAdapter()
    config = BenchmarkAdapterConfig(8, "cosine", 16, 200, 100, 1)
    adapter.build(np.zeros((4, 8), dtype=np.float32), config)
    params = adapter.actual_parameters()
    assert params["dtype"] == "f32"
    assert params["omp_threads"] == 1
