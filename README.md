# VectorForge

CPU ANN retrieval engine with a public, reproducible performance competition framework.

VectorForge is not trying to be a full vector database. The two goals are:

1. Build a high-performance vector retrieval engine.
2. Make every performance change automatically reproducible, measurable, and rejectable when it regresses correctness.

**Current status: Phase 3A complete. Official Phase 3B full-dataset comparison has not been run.** Exact `FlatIndex` is the ground-truth oracle. `HNSWIndex` is a deterministic, paper-faithful ANN index. The Phase 3 harness can compare VectorForge, Faiss, hnswlib, and USearch under matched single-thread conditions. Do not treat smoke numbers as official results.

## What works now

* C++20 library, CMake Debug/Release builds
* Python package (`pip install -e .`)
* Scalar FP32 distances: L2 and cosine
* `FlatIndex` exact search (single-query and batch)
* `HNSWIndex` baseline (`M`, `efConstruction`, `efSearch`, seed)
* FlatIndex save/load (`VF01`) and HNSW save/load (`VH01`)
* Reject non-finite input vectors and queries (`NaN`, `+Inf`, `-Inf`)
* Recall@10 / Recall@100 evaluator vs FlatIndex (`eval/recall.py`)
* Phase 3A fair benchmark infrastructure (`benchmarks/phase3.py`)
* Optional competitor extras (`pip install -e ".[bench]"`)
* C++ unit tests (Catch2) and Python tests
* CI workflow (`.github/workflows/ci.yml`) plus a separate `.[bench]` job

## What is explicitly not in this tree yet

SIMD kernels, quantization, GPU, distributed search, REST, and official competitor performance claims. Empty AVX translation units are placeholders only; they are not used.

Do not read this README as "faster than Faiss / hnswlib / USearch". Those comparisons belong to an official Phase 3B run on a controlled native Linux machine.

## Quick start

### C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"    # tests, ruff, recall gates
# pip install -e ".[bench]"  # Phase 3 adapters + dataset loader
pytest
python eval/correctness.py --metric cosine
python eval/recall.py
```

### Example

```python
from vectorforge import FlatIndex, HNSWIndex
import numpy as np

dim = 32
vectors = np.random.randn(1000, dim).astype(np.float32)
queries = np.random.randn(8, dim).astype(np.float32)

exact = FlatIndex(dim=dim, metric="cosine")
exact.add(vectors)

ann = HNSWIndex(
    dim=dim,
    metric="cosine",
    M=16,
    ef_construction=200,
    ef_search=100,
    seed=42,
)
ann.add(vectors)
ann.ef_search = 100

ids, distances = ann.search(queries, k=10)
```

`ids` and `distances` have shape `(n_queries, k)`. A 1-D query of shape `(dim,)` returns 1-D results. If `k` is larger than the index size, unused slots are `id=-1` and `distance=+inf`.

## Metrics

| metric   | distance returned | nearer means |
|----------|-------------------|--------------|
| `l2`     | Euclidean L2      | smaller      |
| `cosine` | `1 - cosine_similarity` | smaller |

Zero-norm vectors are defined to have cosine similarity `0` with every vector, including other zeros. That matches the NumPy reference in `tests/python/numpy_ref.py`. Ties break toward the smaller id. Non-finite values are rejected instead of being searched or serialized.

## HNSW parameters

| parameter | default | role |
|-----------|---------|------|
| `M` | 16 | max degree on layers `> 0`; layer 0 uses `M0 = 2M` |
| `ef_construction` | 200 | construction search width (`>= M`) |
| `ef_search` | 100 | query search width; effective width is `max(ef_search, k)` |
| `seed` | 42 | PRNG seed for the exponential level distribution |

Same vectors, insertion order, parameters, and seed produce the same graph on a given supported toolchain.

## Project rules

Correctness → Reproducibility → Performance → No Regression.

A faster search that drops recall, hardcodes a dataset, or weakens tests is a failed change. See [CONTRIBUTING.md](CONTRIBUTING.md) and [PERFORMANCE.md](PERFORMANCE.md).

## Documentation

* [BUILDING.md](BUILDING.md) — compilers, CMake options, sanitizers
* [CONTRIBUTING.md](CONTRIBUTING.md) — issue/PR workflow
* [ARCHITECTURE.md](ARCHITECTURE.md) — layout and current components
* [BENCHMARKING.md](BENCHMARKING.md) — how numbers must be produced
* [PERFORMANCE.md](PERFORMANCE.md) — classification and regression rules
* [docs/MASTER_SPEC.md](docs/MASTER_SPEC.md) — project-level specification
* [docs/PHASE3_BENCHMARK_PROTOCOL.md](docs/PHASE3_BENCHMARK_PROTOCOL.md) — Phase 3 fairness rules
* [docs/GITTENSOR.md](docs/GITTENSOR.md) — SN74 registration checklist

## License

MIT. See [LICENSE](LICENSE).
