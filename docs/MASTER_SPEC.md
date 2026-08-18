# VectorForge Master Specification

## Project purpose

VectorForge is not a full vector database.

Its purpose is to build:

1. a high-performance ANN retrieval engine
2. a reproducible performance competition framework

Every accepted improvement must follow this order:

`Correctness -> Reproducibility -> Performance -> No Regression`

## Development phases

### Phase 0

Project foundation: build, tests, packaging, CI, and contributor workflow.

### Phase 1

Exact `FlatIndex` as the correctness oracle. This phase defines the baseline
distance semantics, deterministic tie handling, padding behavior for `k > N`,
and the input-validation rules that future ANN implementations must preserve.

### Phase 2

HNSW baseline. This begins only after Phase 1 is frozen and repeatably green.

### Phase 3

Real embedding benchmark plus competitor harness under matched conditions.

### Phase 4

SIMD optimization, starting from the scalar reference and guarded by
kernel-level correctness checks.

### Phase 5

Quantization and the associated recall / latency / RAM / size trade-offs.

### Phase 6+

Later extensions such as GPU kernels, disk-backed ANN, hybrid dense+sparse
retrieval, and RAG-oriented integrations.

Phases are not skipped. A later phase cannot be treated as complete if an
earlier phase is still unverified.

## SN74 philosophy

VectorForge is designed as a continuous optimization arena. Contributions are
not scored by code volume. They are scored by measured behavior under a shared
evaluator.

Future pull requests should be automatically assessable on metrics such as:

* correctness
* recall
* queries per second
* latency
* memory
* build time

Benchmark gaming is not allowed. The repository must reject fabricated numbers,
dataset-specific shortcuts, weakened tests, altered baselines presented as wins,
or any optimization that bypasses correctness and regression gates.
