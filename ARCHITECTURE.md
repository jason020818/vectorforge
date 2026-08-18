# Architecture

Phase 0/1 implements a small exact-search core. The directory layout matches the project spec, with two documented omissions:

1. **No `hnsw_index.hpp` / `hnsw_index.cpp`.** HNSW is Phase 2. Shipping a stub API would look like a finished feature.
2. **AVX2 / AVX-512 TUs are empty.** They compile so the planned layout stays intact, but `FlatIndex` only calls `src/distance/scalar.cpp`. Illegal-instruction risk is therefore zero on non-AVX CPUs.

## Components

```text
include/vectorforge/   public C++ headers
src/distance/          scalar kernel (active); avx2/avx512 (inactive)
src/index/             FlatIndex brute-force exact search
src/memory/            future alignment / pooling notes
src/threading/         future thread-count helper (unused by FlatIndex)
src/serialization/     reserved for shared binary I/O
python/                pybind11 module + Python package
tests/                 Catch2 + pytest
eval/                  correctness runner; performance/regression later
benchmarks/            local micro-benchmark; competitor adapters later
```

## Search semantics

`Index` is the abstract surface (`add`, `search`, `search_batch`, `save`, `load`). `FlatIndex` stores FP32 vectors in a contiguous `std::vector<float>` with sequential ids `0 .. n-1`.

Top-k uses a bounded max-heap. Ordering is `(distance ascending, id ascending)`.

Cosine distance is `1 - dot(a,b) / (||a|| ||b||)` with similarity defined as `0` when either norm is 0. Accumulators are float64, then stored as float32, matching `tests/python/numpy_ref.py`.

Phase 1 hardening rules:

* `FlatIndex` rejects non-finite values on `add`, `search`, and `search_batch`
* size arithmetic is checked before allocation or copy
* `VF01` loads validate header fields, payload size, and finite vector values before committing state

## Serialization

File format `VF01` (little-endian, Linux x86-64):

```text
magic[4] = 'V' 'F' '0' '1'
u32 kind = 1 (FlatIndex)
u32 dim
u32 metric (0=L2, 1=cosine)
u64 n
float32 vectors[n * dim]
```

## Next phase

Phase 2 adds a paper-faithful HNSW (`M`, `efConstruction`, `efSearch`) that is scored with Recall@10 / Recall@100 against this FlatIndex. SIMD dispatch is after that, not before.
