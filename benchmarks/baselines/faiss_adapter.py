"""Compatibility wrapper for the Phase 3 Faiss adapter."""

from benchmarks.engines.faiss import FaissHnswAdapter


def available() -> bool:
    return FaissHnswAdapter.dependency_available()


def build_and_search(*_args, **_kwargs):
    raise NotImplementedError("Use benchmarks.phase3.py / benchmarks.engines.faiss instead.")
