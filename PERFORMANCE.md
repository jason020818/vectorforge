# Performance policy

Nothing is a win unless correctness still holds.

## Gate

On a named benchmark:

* Recall@10 ≥ configured threshold (once HNSW exists)
* correctness tests PASS
* no crash / no undefined behavior
* no dataset-specific hardcoding
* memory regression within the allowed bound

Then one or more of QPS, P50, P95, index build time, RAM, index size must improve.

## Classification (provisional)

```text
REJECT  correctness or regression failure
NONE    <1% meaningful improvement
XS      1–3%
S       3–7%
M       7–15%
L       15–30%
XL      >30%
```

Do not classify from a single cherry-picked metric. Scoring-rule changes go in version-controlled config, not in a PR description.

## Regression → REJECT even if QPS went up

* Recall below threshold
* correctness failure
* crash or nondeterministic corruption
* memory over the allowance
* serious drop on another representative dataset
* dataset-specific hardcoding
* benchmark manipulation
* baseline protocol change
* tests removed
* evaluator bypass

## Phase 0/1 note

There is no HNSW frontier yet, so CI does not print `eval:M`. The active gate is exact-search correctness. SIMD, FP16, and INT8 are v0.2+ and each needs its own before/after table plus a kernel correctness test.
