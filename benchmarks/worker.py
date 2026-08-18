"""Per-engine isolated worker process for benchmark execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.results import RunConfig, dump_json
from benchmarks.runner import benchmark_engine, exact_ground_truth, load_dataset_for_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    config = RunConfig(**payload["config"])
    dataset = load_dataset_for_run(config.dataset, limit=config.limit, official=config.official)
    metric = payload["metric"]

    if payload.get("ground_truth_artifact"):
        gt_ids = np.load(payload["ground_truth_artifact"])
    elif dataset.metadata.ground_truth_source == "vibe-canonical" and metric == "cosine":
        gt_ids = dataset.ground_truth_ids
    else:
        gt_ids = exact_ground_truth(dataset.vectors, dataset.queries, metric, config.evaluation_k)

    result = benchmark_engine(
        engine_name=args.engine,
        vectors=dataset.vectors,
        queries=dataset.queries,
        gt_ids=gt_ids,
        metric=metric,
        requested_k=config.k,
        evaluation_k=config.evaluation_k,
        M=config.M,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
        warmup=config.warmup,
        repeat=config.repeat,
        threads=config.requested_threads,
        run_label=payload["run_label"],
    )
    dump_json(args.result, result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
