# Architecture

Phase 0–2 implement an exact-search oracle plus a paper-faithful HNSW baseline. SIMD kernels remain unused.

## Components

```text
include/vectorforge/   public C++ headers (Index, FlatIndex, HNSWIndex, distance)
src/distance/          scalar kernel (active); avx2/avx512 (inactive placeholders)
src/index/             FlatIndex exact search; HNSWIndex ANN baseline
src/memory/            future alignment / pooling notes
src/threading/         future thread-count helper (unused)
src/serialization/     reserved for shared binary I/O
python/                pybind11 module + Python package
tests/                 Catch2 + pytest
eval/                  FlatIndex vs NumPy; HNSW vs FlatIndex recall
benchmarks/            local micro-benchmark; competitor adapters later
```

## Search semantics

`Index` is the abstract surface (`add`, `search`, `search_batch`, `save`, `load`). Both indexes store contiguous FP32 vectors with sequential ids `0 .. n-1`. There are no deletions, user ids, or filters in Phase 2.

Top-k ordering is `(distance ascending, id ascending)`. Cosine distance is `1 - cosine_similarity` with similarity `0` when either norm is 0. Accumulators are float64, then stored as float32. HNSW calls the same `distance::compute` kernels as FlatIndex.

Phase 1 hardening rules still apply to both indexes:

* reject non-finite values on `add`, `search`, `search_batch`, and `load`
* size arithmetic is checked before allocation or copy
* failed `load` does not mutate the existing object

## HNSW baseline

`HNSWIndex` follows Malkov & Yashunin (2018):

* degree cap `M` on layers `> 0`, `M0 = 2 * M` on layer 0
* insertion uses `efConstruction`; query search uses `effective_ef = max(efSearch, k)`
* levels: `level = floor(-ln(U) / ln(M))` with `U ∈ (0, 1)` from a seeded `std::mt19937_64`
* neighbor selection is paper Algorithm 4 (`extendCandidates = false`, `keepPrunedConnections = true`)
* construction is sequential and deterministic for the same vectors, order, parameters, and seed on a given toolchain

`graph_digest()` hashes node levels and ordered adjacency lists (no pointers).

## Serialization

### VF01 (FlatIndex)

```text
magic[4] = 'V' 'F' '0' '1'
u32 kind = 1 (FlatIndex)
u32 dim
u32 metric (0=L2, 1=cosine)
u64 n
float32 vectors[n * dim]
```

### VH01 (HNSWIndex)

```text
magic[4] = 'V' 'H' '0' '1'
u32 dim
u32 metric (0=L2, 1=cosine)
u64 n
u64 M
u64 M0
u64 efConstruction
u64 efSearch
u64 seed
i32 max_level          // -1 if empty
i64 entry_point        // -1 if empty
float32 vectors[n * dim]
i32 levels[n]
for each node i, for layer lc in 0 .. levels[i]:
    u32 degree
    i64 neighbors[degree]
```

Both formats are little-endian, exact length, and reject trailing bytes. VF01 and VH01 are not interchangeable.

## Next phase

Phase 3 is a public real-embedding benchmark plus matched competitor harness. SIMD dispatch is after that, not before. AVX translation units remain empty placeholders.
