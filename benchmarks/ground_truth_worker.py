"""Dedicated ground-truth subprocess for smoke benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.results import dump_json
from benchmarks.runner import exact_ground_truth, load_dataset_for_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    dataset = load_dataset_for_run(
        payload["dataset"],
        limit=payload["limit"],
        official=payload["official"],
    )
    gt_ids = exact_ground_truth(
        dataset.vectors,
        dataset.queries,
        payload["metric"],
        payload["ground_truth_k"],
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.result, gt_ids)
    dump_json(
        args.result.with_suffix(".json"),
        {
            "ground_truth_source": "flatindex-exact-subprocess",
            "ground_truth_artifact": str(args.result),
            "ground_truth_k": payload["ground_truth_k"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
