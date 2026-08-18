# VectorForge Agent Rules

`docs/MASTER_SPEC.md` is the top-level specification. If implementation convenience
conflicts with MASTER_SPEC, MASTER_SPEC wins unless the maintainer explicitly
changes the specification.

Before modifying VectorForge:

1. Read `docs/MASTER_SPEC.md`.
2. Read `ARCHITECTURE.md`.
3. Read `BENCHMARKING.md`.
4. Read `PERFORMANCE.md`.
5. Never skip development phases.
6. Never fabricate benchmark numbers.
7. Never weaken correctness tests to improve speed.
8. Never special-case a benchmark dataset.
9. Never claim an unimplemented feature works.
10. Keep changes small and measurable.
11. Run tests before reporting success.
12. Report actual commands and actual results.
