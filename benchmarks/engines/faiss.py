"""Faiss HNSW benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.engines.base import BenchmarkAdapterConfig, package_version


class FaissHnswAdapter:
    name = "faiss"

    def __init__(self) -> None:
        self.index = None
        self._module = None
        self._config: BenchmarkAdapterConfig | None = None

    @staticmethod
    def dependency_available() -> bool:
        try:
            import faiss  # noqa: F401
        except ImportError:
            return False
        return True

    def _require(self):
        if self._module is None:
            import faiss

            self._module = faiss
        return self._module

    def build(self, vectors: np.ndarray, config: BenchmarkAdapterConfig) -> None:
        self._config = config
        faiss = self._require()
        faiss.omp_set_num_threads(config.threads)
        metric = faiss.METRIC_L2 if config.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index = faiss.IndexHNSWFlat(config.dim, config.M, metric)
        index.hnsw.efConstruction = config.ef_construction
        index.hnsw.efSearch = config.ef_search
        xb = np.ascontiguousarray(vectors, dtype=np.float32)
        if config.metric == "cosine":
            xb = xb.copy()
            faiss.normalize_L2(xb)
        index.add(xb)
        self.index = index

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray | None]:
        if self.index is None:
            raise RuntimeError("index has not been built")
        faiss = self._require()
        xq = np.ascontiguousarray(queries, dtype=np.float32)
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            xq = xq.copy()
            faiss.normalize_L2(xq)
        distances, ids = self.index.search(xq, k)
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            distances = 1.0 - distances
        return np.asarray(ids), np.asarray(distances)

    def set_search_ef(self, ef: int) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        self.index.hnsw.efSearch = ef

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        faiss = self._require()
        faiss.write_index(self.index, str(path))

    def index_size_bytes(self, path: Path) -> int | None:
        return path.stat().st_size if path.exists() else None

    def version_info(self) -> dict[str, str]:
        return {"faiss-cpu": package_version("faiss-cpu", "faiss")}

    def actual_parameters(self) -> dict[str, object]:
        if self.index is None or self._config is None:
            raise RuntimeError("index has not been built")
        return {
            "index_type": type(self.index).__name__,
            "metric_type": "METRIC_L2" if self._config.metric == "l2" else "METRIC_INNER_PRODUCT",
            "normalization_policy": "l2-normalize queries and vectors for cosine"
            if self._config.metric == "cosine"
            else "none",
            "M": self._config.M,
            "efConstruction": self._config.ef_construction,
            "efSearch": self._config.ef_search,
            "omp_threads": self._config.threads,
            "dtype": "f32",
        }

    def effective_threads(self) -> str:
        if self._config is None:
            raise RuntimeError("index has not been built")
        return str(self._config.threads)
