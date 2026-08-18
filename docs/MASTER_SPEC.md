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

Deterministic paper-faithful `HNSWIndex` baseline (`M`, `efConstruction`, `efSearch`, seed),
L2 and cosine, batch search, `VH01` save/load, and Recall@10 / Recall@100 against
FlatIndex on a documented synthetic set. This phase is correctness and
reproducibility, not SIMD or competitor QPS.

### Phase 3

Real embedding benchmark plus competitor harness under matched conditions.

Phase 3A (infrastructure, smoke, fairness hardening) is complete. Phase 3B is the official full-dataset run on a controlled native Linux machine. It has not started.

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
* VF01 loads validate magic, kind, dim, metric, size arithmetic, exact payload length, and finite vector values, and do not commit object state on failure
* VF01 payload length must match exactly. Trailing bytes are invalid. Future serialization extensions must use a separately specified version or format rather than unknown trailing data.

## Phase 2 requirements

Implemented:

* `HNSWIndex`
* parameters `M`, `efConstruction`, `efSearch`, `seed`
* deterministic construction via seeded `std::mt19937_64` level generation
* L2 and cosine (same kernels as FlatIndex)
* batch search
* `VH01` save / load
* Recall@10 and Recall@100 against FlatIndex ground truth (`eval/recall.py`)

## Phase 3 requirements

Phase 3A is implemented:

* one public real embedding dataset: VIBE `ccnews-nomic-768-normalized`
* matched conditions across engines (dataset, queries, metric, top-k, threads, warmup, repetitions)
* baseline adapters for VectorForge, Faiss, hnswlib, and USearch
* reported fields: QPS, P50, P95, P99, peak RAM, build time, serialized index size, plus recall
* official mode hard-fails unless `OFFICIAL_ENVIRONMENT_READY=true`

Phase 3B is the official full-dataset comparison. It is not complete until a native Linux run records `run_label = OFFICIAL`.

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
