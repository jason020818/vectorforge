"""USearch benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.engines.base import BenchmarkAdapterConfig, package_version


class USearchAdapter:
    name = "usearch"

    def __init__(self) -> None:
        self.index = None
        self._module = None
        self._config: BenchmarkAdapterConfig | None = None
        self._thread_control_status = "not_directly_observable"

    @staticmethod
    def dependency_available() -> bool:
        try:
            from usearch.index import Index  # noqa: F401
        except ImportError:
            return False
        return True

    def _require(self):
        if self._module is None:
            import usearch.index as usearch_index

            self._module = usearch_index
        return self._module

    def build(self, vectors: np.ndarray, config: BenchmarkAdapterConfig) -> None:
        self._config = config
        usearch_index = self._require()
        metric = "l2sq" if config.metric == "l2" else "cos"
        index = usearch_index.Index(
            ndim=config.dim,
            metric=metric,
            dtype="f32",
            connectivity=config.M,
            expansion_add=config.ef_construction,
            expansion_search=config.ef_search,
        )
        xb = np.ascontiguousarray(vectors, dtype=np.float32)
        index.add(np.arange(xb.shape[0]), xb, threads=config.threads, dtype="f32")
        self._thread_control_status = "explicit_threads_parameter"
        self.index = index

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray | None]:
        if self.index is None:
            raise RuntimeError("index has not been built")
        xq = np.ascontiguousarray(queries, dtype=np.float32)
        matches = self.index.search(xq, count=k, threads=self._config.threads, dtype="f32")
        return np.asarray(matches.keys), np.asarray(matches.distances)

    def set_search_ef(self, ef: int) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        self.index.expansion_search = ef

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        self.index.save(str(path))

    def index_size_bytes(self, path: Path) -> int | None:
        return path.stat().st_size if path.exists() else None

    def version_info(self) -> dict[str, str]:
        return {"usearch": package_version("usearch")}

    def actual_parameters(self) -> dict[str, object]:
        if self._config is None:
            raise RuntimeError("index has not been built")
        return {
            "metric": "l2sq" if self._config.metric == "l2" else "cos",
            "dtype": "f32",
            "connectivity": self._config.M,
            "expansion_add": self._config.ef_construction,
            "expansion_search": self._config.ef_search,
            "thread_control_status": self._thread_control_status,
        }

    def effective_threads(self) -> str:
        if self._config is None:
            raise RuntimeError("index has not been built")
        return str(self._config.threads)
