# Phase 3A Validation

Phase 3A fairness hardening validation.

## Provenance

- Base commit: `9726efd19460774ef21681d9be69017f02731ead`
- Validated code/test commit: `af58538d6306bdb64d11bc4f00fdae0751321ec3`
- Working tree during validation: clean
- Verified dataset SHA256:
  `c3a498239eea772b7b736f31c25f8dff089f349352ec523b0d9cb694c71f1df3`

## Environment

- OS: `Linux`
- Kernel: `6.18.33.2-microsoft-standard-WSL2`
- CPU: `Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz`
- Python: `3.12.3`
- Faiss: `1.15.0`
- hnswlib: `0.8.0`
- USearch: `2.26.0`

## Tests

- C++ Release: `75/75 passed`
- C++ Debug + ASan/UBSan: `75/75 passed`
- Python `.[dev]` clean environment: `47 passed, 3 skipped`
- Benchmark `.[bench]` integration: `27 passed`
- Ruff: `PASS`
- clang-format: `PASS`

## Smoke Configuration

- Dataset: `ccnews-nomic-768-normalized`
- Database limit: `10000`
- Query count: `1000`
- Metric: `cosine`
- `M`: `16`
- `efConstruction`: `200`
- `efSearch`: `100`
- Threads: `1`
- Warmup: `2`
- Repeat: `3`
- Ground truth source: `flatindex-exact-subprocess`
- Ground truth artifact:
  `benchmarks/results/ccnews-nomic-768-normalized-20260818T092218Z/ground_truth.npy`

## Smoke Results

NON-OFFICIAL SMOKE RESULT

- VectorForge: `Recall@10=0.9969`, `Recall@100=0.97332`
- Faiss: `Recall@10=0.9974`, `Recall@100=0.97238`
- hnswlib: `Recall@10=0.9965`, `Recall@100=0.96696`
- USearch: `Recall@10=0.9965`, `Recall@100=0.96706`

Results directory:

`benchmarks/results/ccnews-nomic-768-normalized-20260818T092218Z`

## Status

PHASE 3A STATUS: COMPLETE

READY FOR PHASE 3B: YES
