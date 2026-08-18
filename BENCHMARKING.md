# Benchmarking

## Rule

A number that was not produced by a command in this repository, on a named commit, is not a project result. Do not paste guessed QPS into README, issues, or PRs.

## Phase 0/1

The only honest measurements today are:

* unit tests pass / fail
* NumPy vs FlatIndex id agreement
* optional local `python benchmarks/benchmark.py` micro-timings for sanity

`benchmarks/benchmark.py` prints a median over repeats. Those timings are **not** a published claim and must not be copied into README.

Competitor adapters (`faiss`, `hnswlib`, `usearch`) raise `NotImplementedError` until HNSW exists. Comparing exact scan to graph indexes would not be a fair harness.

## After HNSW

The harness must lock:

* dataset and query set
* metric and top-k
* thread count / CPU affinity
* warmup and repetition count
* statistic (median or another named robust stat)

Reported fields:

```text
Recall@10
Recall@100
Queries/sec
P50 latency
P95 latency
P99 latency
Index build time
Peak RSS
Serialized index size
```

Recall is computed against FlatIndex (or an equivalent exact dump from `benchmarks/ground_truth.py`). A run with Recall@10 below the stated threshold is not a performance winner, regardless of QPS.

## Datasets

Local development may use small synthetic vectors. Any performance claim that leaves this repository must use a public real embedding set (on the order of 1M × 768–1024d). One dataset is fixed first; more datasets are added only after the evaluator is stable.

## Anti-gaming

Do not:

* drop repetitions
* change the baseline measurement method in the same PR as the "win"
* hardcode results for a named dataset
* disable correctness checks
* compare against a crippled competitor configuration
