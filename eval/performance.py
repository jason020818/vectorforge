"""Placeholder performance evaluator.

Phase 0/1 does not claim QPS/latency numbers. Performance comparison against
Faiss/hnswlib/USearch starts after the HNSW baseline exists.
"""

from __future__ import annotations


def main() -> int:
    print("performance.py: not active in Phase 0/1 (no HNSW baseline yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
