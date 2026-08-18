"""Compatibility wrapper for the Phase 3 hnswlib adapter."""

from benchmarks.engines.hnswlib import HnswlibAdapter


def available() -> bool:
    return HnswlibAdapter.dependency_available()


def build_and_search(*_args, **_kwargs):
    raise NotImplementedError("Use benchmarks.phase3.py / benchmarks.engines.hnswlib instead.")
