"""VectorForge HNSW benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from vectorforge import HNSWIndex, __version__

from benchmarks.engines.base import BenchmarkAdapterConfig


class VectorForgeAdapter:
    name = "vectorforge"

    def __init__(self) -> None:
        self.index: HNSWIndex | None = None
        self._config: BenchmarkAdapterConfig | None = None

    @staticmethod
    def dependency_available() -> bool:
        return True

    def build(self, vectors: np.ndarray, config: BenchmarkAdapterConfig) -> None:
        self._config = config
        self.index = HNSWIndex(
            dim=config.dim,
            metric=config.metric,
            M=config.M,
            ef_construction=config.ef_construction,
            ef_search=config.ef_search,
            seed=42,
        )
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray | None]:
        if self.index is None:
            raise RuntimeError("index has not been built")
        ids, distances = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k=k)
        return np.asarray(ids), np.asarray(distances)

    def set_search_ef(self, ef: int) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        self.index.ef_search = ef

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built")
        self.index.save(str(path))

    def index_size_bytes(self, path: Path) -> int | None:
        return path.stat().st_size if path.exists() else None

    def version_info(self) -> dict[str, str]:
        return {"vectorforge": __version__}

    def actual_parameters(self) -> dict[str, object]:
        if self.index is None or self._config is None:
            raise RuntimeError("index has not been built")
        return {
            "metric": self._config.metric,
            "M": self._config.M,
            "M0": 2 * self._config.M,
            "efConstruction": self._config.ef_construction,
            "efSearch": self._config.ef_search,
            "seed": 42,
            "dtype": "f32",
            "threads": self._config.threads,
        }

    def effective_threads(self) -> str:
        if self._config is None:
            raise RuntimeError("index has not been built")
        return str(self._config.threads)
