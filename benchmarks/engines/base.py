"""Common engine adapter interface for Phase 3 benchmarks."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class BenchmarkAdapterConfig:
    dim: int
    metric: str
    M: int
    ef_construction: int
    ef_search: int
    threads: int


class EngineAdapter(Protocol):
    name: str

    @staticmethod
    def dependency_available() -> bool: ...

    def build(self, vectors: np.ndarray, config: BenchmarkAdapterConfig) -> None: ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray | None]: ...

    def set_search_ef(self, ef: int) -> None: ...

    def save(self, path: Path) -> None: ...

    def index_size_bytes(self, path: Path) -> int | None: ...

    def version_info(self) -> dict[str, str]: ...

    def actual_parameters(self) -> dict[str, object]: ...

    def effective_threads(self) -> str: ...


def package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def create_engine(name: str) -> EngineAdapter:
    if name == "vectorforge":
        from benchmarks.engines.vectorforge import VectorForgeAdapter

        return VectorForgeAdapter()
    if name == "faiss":
        from benchmarks.engines.faiss import FaissHnswAdapter

        return FaissHnswAdapter()
    if name == "hnswlib":
        from benchmarks.engines.hnswlib import HnswlibAdapter

        return HnswlibAdapter()
    if name == "usearch":
        from benchmarks.engines.usearch import USearchAdapter

        return USearchAdapter()
    raise ValueError(f"unknown benchmark engine '{name}'")
