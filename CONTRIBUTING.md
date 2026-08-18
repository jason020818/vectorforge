# Contributing

VectorForge treats a contribution as a measured change, not as lines of code.

## Path for a new contributor

```text
clone → build → tests → benchmark → pick an issue → patch → local eval → PR
```

1. Follow [BUILDING.md](BUILDING.md) until C++ tests and `pytest` pass.
2. Run `python eval/correctness.py` (and, after HNSW exists, the performance harness).
3. Pick an issue that states a baseline, a target, and constraints.
4. Do not start HNSW or SIMD work until Phase 1 (exact FlatIndex) stays green.

## Issue style

Write a measurable engineering challenge:

```text
PERF-001
Optimize cosine distance using AVX2

Baseline:
<measured QPS on named commit + dataset>

Target:
>= 5% QPS improvement

Constraints:
Recall unchanged
Peak memory regression < 2%
AVX2-unsupported CPU must fall back safely
All unit tests must pass
```

"Make search faster" is not an acceptable issue.

## Pull requests

* Keep PRs small.
* Do not delete or weaken tests to get CI green.
* Do not claim benchmark numbers you did not run.
* Do not special-case a dataset id.
* If you change a distance kernel, add/keep a scalar vs new-kernel correctness test.

CI must pass Debug and Release C++ tests plus Python correctness tests.

## Coding standards

* C++20, `-Wall -Wextra -Wpedantic -Werror`
* Google-based `.clang-format`
* Python: ruff
* One test framework: Catch2 (do not add GoogleTest)

## What not to do

See the project spec: no fabricated numbers, no hardcoded answers, no disabling features for speed, no evaluator bypass, no treating a stub as a production kernel.
