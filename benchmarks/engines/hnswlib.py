"""hnswlib benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.engines.base import BenchmarkAdapterConfig, package_version


class HnswlibAdapter:
    name = "hnswlib"

    def __init__(self) -> None:
        self.index = None
        self._module = None
        self._config: BenchmarkAdapterConfig | None = None

    @staticmethod
    def dependency_available() -> bool:
        try:
            import hnswlib  # noqa: F401
        except ImportError:
            return False
        return True

    def _require(self):
        if self._module is None:
            import hnswlib

            self._module = hnswlib
        return self._module

    def build(self, vectors: np.ndarray, config: BenchmarkAdapterConfig) -> None:
        self._config = config
        hnswlib = self._require()
        space = "l2" if config.metric == "l2" else "cosine"
        index = hnswlib.Index(space=space, dim=config.dim)
        index.init_index(
            max_elements=int(vectors.shape[0]),
            M=config.M,
            ef_construction=config.ef_construction,
            random_seed=42,
        )
        index.set_num_threads(config.threads)
        index.set_ef(config.ef_search)
        index.add_items(
            np.ascontiguousarray(vectors, dtype=np.float32),
            np.arange(vectors.shape[0]),
            num_threads=config.threads,
        )
        self.index = index

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray | None]:
        if self.index is None:
            raise RuntimeError("index has not been built")
        labels, distances = self.index.knn_query(
            np.ascontiguousarray(queries, dtype=np.float32),
            k=k,
            num_threads=self._config.threads if self._config is not None else -1,
        )
        return np.asarray(labels), np.asarray(distances)

    def set_search_ef(self, ef: int) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        self.index.set_ef(ef)

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        self.index.save_index(str(path))

    def index_size_bytes(self, path: Path) -> int | None:
        return path.stat().st_size if path.exists() else None

    def version_info(self) -> dict[str, str]:
        return {"hnswlib": package_version("hnswlib")}

    def actual_parameters(self) -> dict[str, object]:
        if self._config is None:
            raise RuntimeError("index has not been built")
        return {
            "space": "l2" if self._config.metric == "l2" else "cosine",
            "M": self._config.M,
            "ef_construction": self._config.ef_construction,
            "ef": self._config.ef_search,
            "random_seed": 42,
            "num_threads": self._config.threads,
            "dtype": "f32",
        }

    def effective_threads(self) -> str:
        if self._config is None:
            raise RuntimeError("index has not been built")
        return str(self._config.threads)
