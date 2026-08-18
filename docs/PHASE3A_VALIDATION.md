# Phase 3A Validation

Phase 3A fairness hardening validation.

NON-OFFICIAL SMOKE RESULT only.

## Provenance

- Base commit: `9726efd19460774ef21681d9be69017f02731ead`
- Validated source tree: local fairness-hardening working tree atop the base commit
- Working-tree state during validation: dirty
- Validation record commit: not created locally because Git author identity is not configured

## Environment

- Python: `3.12.3`
- OS: `Linux`
- Kernel: `6.18.33.2-microsoft-standard-WSL2`
- CPU: `Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz`
- Faiss: `1.15.0`
- hnswlib: `0.8.0`
- USearch: `2.26.0`

## Verification

- `pytest`: `54 passed, 1 skipped`
- Phase 3A benchmark-dependency tests: `24 passed`
- `ruff check python tests/python eval benchmarks`: `PASS`

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
  `benchmarks/results/ccnews-nomic-768-normalized-20260818T090245Z/ground_truth.npy`

## Smoke Results

NON-OFFICIAL SMOKE RESULT

- VectorForge: `Recall@10=0.9969`, `Recall@100=0.97332`
- Faiss: `Recall@10=0.9974`, `Recall@100=0.97238`
- hnswlib: `Recall@10=0.9965`, `Recall@100=0.96696`
- USearch: `Recall@10=0.9965`, `Recall@100=0.96706`

Results directory:

`benchmarks/results/ccnews-nomic-768-normalized-20260818T090245Z`
