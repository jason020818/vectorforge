"""Compatibility wrapper for the Phase 3 USearch adapter."""

from benchmarks.engines.usearch import USearchAdapter


def available() -> bool:
    return USearchAdapter.dependency_available()


def build_and_search(*_args, **_kwargs):
    raise NotImplementedError("Use benchmarks.phase3.py / benchmarks.engines.usearch instead.")
