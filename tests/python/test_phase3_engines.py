from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from benchmarks.engines import BenchmarkAdapterConfig, create_engine


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
