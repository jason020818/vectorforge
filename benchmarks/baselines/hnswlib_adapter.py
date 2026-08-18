"""hnswlib baseline adapter. Enabled after the HNSW baseline exists."""


def available() -> bool:
    return False


def build_and_search(*_args, **_kwargs):
    raise NotImplementedError(
        "hnswlib comparison is part of the benchmark harness after HNSW (Phase 2+)."
    )
