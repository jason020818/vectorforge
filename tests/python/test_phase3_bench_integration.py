from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.engines import BenchmarkAdapterConfig, create_engine
from benchmarks.results import recall_at_k


@pytest.mark.skipif(
    os.environ.get("VECTORFORGE_BENCH_INTEGRATION") != "1",
    reason="benchmark dependency integration job only",
)
def test_benchmark_dependency_integration() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((64, 16), dtype=np.float32)
    queries = rng.standard_normal((8, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    config = BenchmarkAdapterConfig(
        dim=16,
        metric="cosine",
        M=16,
        ef_construction=200,
        ef_search=100,
        threads=1,
    )
    ids_by_engine: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for name in ("vectorforge", "faiss", "hnswlib", "usearch"):
            adapter = create_engine(name)
            assert adapter.dependency_available()
            adapter.build(vectors, config)
            ids, distances = adapter.search(queries, 10)
            ids = np.asarray(ids)
            ids_by_engine[name] = ids
            assert ids.shape == (8, 10)
            if distances is not None:
                assert np.asarray(distances).shape == (8, 10)
            artifact = Path(temp_dir) / f"{name}.index"
            adapter.save(artifact)
            assert adapter.index_size_bytes(artifact) is not None
            assert adapter.effective_threads() in {"1", "not_directly_observable"}

    gt = ids_by_engine["vectorforge"]
    for name, ids in ids_by_engine.items():
        score = recall_at_k(ids.tolist(), gt.tolist(), 10)
        assert score > 0.8, f"{name} recall too low in integration fixture: {score}"
