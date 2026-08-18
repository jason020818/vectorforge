# VectorForge

CPU ANN retrieval engine with a public, reproducible performance competition framework.

VectorForge is not trying to be a full vector database. The two goals are:

1. Build a high-performance vector retrieval engine.
2. Make every performance change automatically reproducible, measurable, and rejectable when it regresses correctness.

**Current status: v0.1 foundation (Phase 0 + Phase 1).** Exact `FlatIndex` is implemented. HNSW is not.

## What works now

* C++20 library, CMake Debug/Release builds
* Python package (`pip install -e .`)
* Scalar FP32 exact search: L2 and cosine
* Single-query and batch search
* FlatIndex save/load
* C++ unit tests (Catch2)
* Python correctness tests vs NumPy brute-force
* GitHub Actions CI

## What is explicitly not in this tree yet

HNSW, SIMD kernels, quantization, GPU, distributed search, REST, and competitor leaderboards. Those start after this exact-search baseline is green. Empty AVX translation units are placeholders only; they are not used.

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
pip install -e ".[dev]"
pytest
python eval/correctness.py --metric cosine
```

### Example

```python
from vectorforge import FlatIndex
import numpy as np

dim = 32
index = FlatIndex(dim=dim, metric="cosine")
index.add(np.random.randn(1000, dim).astype(np.float32))

queries = np.random.randn(8, dim).astype(np.float32)
ids, distances = index.search(queries, k=10)
```

`ids` and `distances` have shape `(n_queries, k)`. A 1-D query of shape `(dim,)` returns 1-D results. If `k` is larger than the index size, unused slots are `id=-1` and `distance=+inf`.

## Metrics

| metric   | distance returned | nearer means |
|----------|-------------------|--------------|
| `l2`     | Euclidean L2      | smaller      |
| `cosine` | `1 - cosine_similarity` | smaller |

Zero-norm vectors are defined to have cosine similarity `0` with every vector, including other zeros. That matches the NumPy reference in `tests/python/numpy_ref.py`. Ties break toward the smaller id.

## Project rules

Correctness → Reproducibility → Performance → No Regression.

A faster search that drops recall, hardcodes a dataset, or weakens tests is a failed change. See [CONTRIBUTING.md](CONTRIBUTING.md) and [PERFORMANCE.md](PERFORMANCE.md).

## Documentation

* [BUILDING.md](BUILDING.md) — compilers, CMake options, sanitizers
* [CONTRIBUTING.md](CONTRIBUTING.md) — issue/PR workflow
* [ARCHITECTURE.md](ARCHITECTURE.md) — layout and current components
* [BENCHMARKING.md](BENCHMARKING.md) — how numbers must be produced
* [PERFORMANCE.md](PERFORMANCE.md) — classification and regression rules

## License

MIT. See [LICENSE](LICENSE).
