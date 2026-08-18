# Phase 3 Benchmark Protocol

Phase 3 introduces the first real-embedding benchmark infrastructure for VectorForge.

This document defines how Phase 3 benchmark runs must be produced and interpreted.

## Dataset

The first official Phase 3 dataset is VIBE `ccnews-nomic-768-normalized`.

Canonical source:

- Hugging Face dataset repo: `vector-index-bench/vibe`
- File: `ccnews-nomic-768-normalized.hdf5`
- Reference commit: `ca58b61673bb8e9ff5cf2279668161a93ebfbdab`

The dataset is public, real-embedding, and large enough to replace synthetic Phase 2-only validation for benchmark infrastructure.

The benchmark adapter must:

- download or locate the canonical file
- cache it locally
- compute and verify the actual SHA256 against the expected canonical digest
- preserve float32 vectors
- reject malformed or non-finite data
- record source metadata and dataset identity

The dataset file itself must not be committed to Git.

## Fairness Rules

Phase 3A establishes a single-thread baseline because VectorForge Phase 2 construction and search are currently single-thread oriented.

The benchmark runner must therefore:

- request one CPU thread for every engine
- set relevant thread-control environment variables such as `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` in the subprocess environment before worker Python startup
- record requested thread count, observable affinity, and environment metadata
- avoid unsupported claims that a library is single-threaded when that cannot be directly observed

The first comparison is parameter-matched HNSW:

- `M = 16`
- `efConstruction = 200`
- `efSearch = 100`

If a library uses different parameter names, the adapter must record the mapping explicitly.

## Recall

Recall is computed from IDs, not from approximate distances.

Definition:

```text
Recall@K =
mean over queries(
  |ANN top-K IDs ∩ ground-truth top-K IDs| / K
)
```

Canonical ground truth should be used when present and validated.

Subset smoke runs must compute exact ground truth once in a dedicated `FlatIndex` subprocess and persist a shared artifact that every engine worker reads.

## Warmup and Repetition

Warmup queries are never included in measured samples.

Smoke runs may use reduced repetitions for pipeline validation.

Official runs require at least:

- `warmup >= 2`
- `repeat >= 5`

The main reported repeated-run statistic is the median. Individual run values remain in machine-readable output.

Best-run cherry-picking is not allowed.

## Latency and QPS

Latency percentiles must be computed from per-query samples, not from one giant batch wall-clock time.

QPS must:

- use the same query workload for every engine
- exclude build time
- be interpreted together with recall

Do not publish a QPS value without the matching recall values.

## Memory and Index Size

Peak RSS must be measured in the isolated engine process. Where practical, record:

- baseline process RSS
- post-build RSS
- peak RSS

For smoke runs that need exact ground truth, that calculation must happen in a separate subprocess so engine `peak RSS` excludes `FlatIndex` oracle work.

Serialized index size must use a real file:

- VectorForge: `VH01`
- competitor engines: native format when supported

If an adapter cannot serialize, record the size as unavailable instead of inventing it.

## Smoke vs Official

Smoke runs are allowed for development and pipeline validation.

Typical smoke mode:

```text
--limit 10000
```

Smoke output must be labeled:

```text
NON-OFFICIAL SMOKE RESULT
```

Only full-dataset runs may be official.

`--official` must reject `--limit`.

## Environment Requirements

Environment preflight records:

- CPU model
- logical and physical CPU counts when available
- CPU affinity
- OS and kernel
- WSL detection
- SMT/hyperthreading state when available
- CPU governor when available
- transparent huge page state when available
- RAM

If official Linux benchmark controls are not established, the runner should set:

```text
OFFICIAL_ENVIRONMENT_READY = false
```

Uncontrolled environments may still run smoke benchmarks.

`--official` must hard fail when `OFFICIAL_ENVIRONMENT_READY` is false. A non-official full-dataset run may be labeled `NON-OFFICIAL / UNCONTROLLED`, but never `OFFICIAL`.

## Result Schema

Benchmark runs write deterministic JSON files under:

```text
benchmarks/results/<run-id>/
```

Expected artifacts include:

- `environment.json`
- `dataset.json`
- one result JSON per engine
- `summary.json`

Markdown comparison tables must be generated from these JSON files, not manually typed into README.

## Anti-Gaming Rules

Do not:

- swap in a synthetic replacement for the official dataset
- weaken recall thresholds
- change dataset/query workload across engines
- silently use all CPU cores for competitors
- publish smoke numbers as official benchmark results
- report best-run-only numbers
- estimate index size or memory from Python object sizes
- modify VectorForge algorithm behavior to improve the first Phase 3 result

Phase 3A ends after infrastructure, smoke validation, and tests. It does not publish an official full comparison yet.
