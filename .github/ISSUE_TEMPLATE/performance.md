---
name: Performance Challenge
about: Propose a measured performance change. Do not open one without a recorded baseline.
labels: enhancement
---

## Identifier

<!-- Example: PERF-001 -->

## Baseline

- Commit SHA:
- Dataset:
- Hardware / OS:
- Metric / k / M / efConstruction / efSearch / threads:
- Command actually run:
- Measured QPS / latency / recall / RSS:

Do not invent baseline numbers. If Phase 3B official numbers are not published yet, do not open a competitor-QPS issue.

## Target

<!-- Example: >= 5% QPS improvement on the same official protocol. -->

## Constraints

- Recall must not regress below the stated threshold.
- Peak memory regression must stay within the stated bound.
- All unit tests, sanitizers, and recall gates must remain green.
- AVX-unsupported CPUs must fall back safely if SIMD is involved.
- Do not special-case a dataset id.
- Do not weaken tests or change the measurement protocol in the same PR as the optimization.

## Acceptance command

```text
# exact command that must be re-run after the change
```
