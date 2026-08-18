# VectorForge Master Specification

## Project purpose

VectorForge is not a full vector database.

Its purpose is to build:

1. a high-performance ANN retrieval engine
2. a reproducible performance competition framework

Every accepted improvement must follow this order:

`Correctness -> Reproducibility -> Performance -> No Regression`

## Required technology

Phase 1 and later phases keep this core stack unless the maintainer changes it:

* C++20
* CMake
* pybind11
* Catch2
* Python correctness / reference harness (NumPy brute-force plus `eval/correctness.py`)

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

## Phase 1 invariants

After Phase 1 freeze, changing any of the following requires an explicit
compatibility justification in the PR, not a silent edit:

* L2 is Euclidean distance
* cosine distance is `1 - cosine_similarity`
* zero-norm vectors have cosine similarity `0` with every vector, including other zeros
* equal distances break ties toward the smaller id
* when `k > N`, unused slots are `id = -1` and `distance = +inf`
* `add`, `search`, `search_batch`, and `load` reject non-finite values (`NaN`, `+Inf`, `-Inf`)
* VF01 loads validate magic, kind, dim, metric, size arithmetic, payload length, and finite vector values, and do not commit object state on failure

## Phase 2 requirements

Not implemented in Phase 1. Specification only:

* `HNSWIndex`
* parameters `M`, `efConstruction`, `efSearch`
* deterministic / reproducible construction policy
* L2 and cosine
* batch search
* save / load
* Recall@10 and Recall@100 against FlatIndex ground truth

## Phase 3 requirements

Not implemented in Phase 1. Specification only:

* one public real embedding dataset first
* matched conditions across engines (dataset, queries, metric, top-k, threads, warmup, repetitions)
* baseline adapters for Faiss, hnswlib, and USearch
* VIBE-compatible direction after the evaluator is stable
* reported fields: QPS, P50, P95, P99, peak RAM, build time, serialized index size, plus recall

## Anti-gaming rules

The following are never a performance win:

* dataset-specific hardcoding
* weakened correctness threshold
* reduced benchmark repetitions
* crippled or otherwise unfair competitor settings
* test deletion or weakening to pass CI
* changing the baseline measurement protocol in the same change as an "optimization"
* fabricated results
* hidden feature disablement for speed

## SN74 contribution philosophy

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

Each performance issue should include, when possible:

```text
Baseline
Target
Dataset
Hardware/environment
Correctness constraints
Recall threshold
Memory constraint
Acceptance command
```

Benchmark gaming is not allowed. The repository must reject fabricated numbers,
dataset-specific shortcuts, weakened tests, altered baselines presented as wins,
or any optimization that bypasses correctness and regression gates.
