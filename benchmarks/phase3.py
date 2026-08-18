"""Phase 3 benchmark runner with isolated engine processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.environment import (
    collect_environment,
    ensure_official_mode_allowed,
    environment_warnings_lines,
)
from benchmarks.results import RunConfig, dump_json, summarize_results
from benchmarks.runner import (
    effective_metric,
    load_dataset_for_run,
    run_ground_truth_subprocess,
    run_worker_subprocess,
)

ENGINES = ("vectorforge", "faiss", "hnswlib", "usearch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="append", required=True, help="Engine name or 'all'")
    parser.add_argument(
        "--dataset", choices=["ccnews-nomic-768-normalized"], default="ccnews-nomic-768-normalized"
    )
    parser.add_argument("--metric", choices=["l2", "cosine"], default="cosine")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--allow-uncontrolled-environment", action="store_true")
    parser.add_argument("--results-root", type=Path, default=Path("benchmarks/results"))
    return parser.parse_args()


def _expand_engines(raw: list[str]) -> list[str]:
    if "all" in raw:
        return list(ENGINES)
    engines = []
    for name in raw:
        if name not in ENGINES:
            raise ValueError(f"unknown engine '{name}'")
        engines.append(name)
    return engines


def failure_result_payload(
    *,
    engine_name: str,
    metric: str,
    requested_k: int,
    evaluation_k: int,
    run_label: str,
    error: str,
) -> dict[str, object]:
    return {
        "engine": engine_name,
        "available": False,
        "dependency_available": False,
        "error": error,
        "metric": metric,
        "requested_k": requested_k,
        "evaluation_k": evaluation_k,
        "run_label": run_label,
    }


def main() -> int:
    args = _parse_args()
    if args.limit is not None and args.official:
        raise SystemExit("--official cannot be combined with --limit")
    if args.k < 10:
        raise SystemExit("k must be at least 10 because Recall@10 is always reported")
    if args.official and (args.warmup < 2 or args.repeat < 5):
        raise SystemExit("official mode requires warmup >= 2 and repeat >= 5")

    environment = collect_environment()
    ensure_official_mode_allowed(
        environment,
        official=args.official,
        allow_uncontrolled_environment=args.allow_uncontrolled_environment,
    )
    for line in environment_warnings_lines(environment):
        print(line)

    dataset = load_dataset_for_run(args.dataset, limit=args.limit, official=args.official)
    metric = effective_metric(args.dataset, args.metric)
    engines = _expand_engines(args.engine)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if args.official:
        run_label = "OFFICIAL"
    elif args.limit is not None:
        run_label = "NON-OFFICIAL SMOKE RESULT"
    elif environment.official_environment_ready:
        run_label = "NON-OFFICIAL"
    else:
        run_label = "NON-OFFICIAL / UNCONTROLLED"
    run_id = f"{args.dataset}-{timestamp}"
    results_dir = args.results_root / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        dataset=args.dataset,
        engine="all" if len(engines) > 1 else engines[0],
        metric=metric,
        k=args.k,
        evaluation_k=max(100, args.k),
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        warmup=args.warmup,
        repeat=args.repeat,
        limit=args.limit,
        official=args.official,
        requested_threads=1,
        allow_uncontrolled_environment=args.allow_uncontrolled_environment,
        git_commit=git_commit,
        timestamp_utc=timestamp,
        results_dir=str(results_dir),
    )
    ground_truth_artifact = None
    if dataset.metadata.ground_truth_source == "vibe-canonical" and metric == "cosine":
        dataset.metadata.ground_truth_source = "vibe-canonical-validated"
        dataset.metadata.ground_truth_k = max(100, args.k)
    else:
        ground_truth_artifact = results_dir / "ground_truth.npy"
        gt_payload = {
            "dataset": args.dataset,
            "limit": args.limit,
            "official": args.official,
            "metric": metric,
            "ground_truth_k": max(100, args.k),
        }
        gt_payload_path = results_dir / "ground_truth_payload.json"
        dump_json(gt_payload_path, gt_payload)
        gt_proc = run_ground_truth_subprocess(
            payload_path=gt_payload_path,
            result_path=ground_truth_artifact,
        )
        if gt_proc.returncode != 0:
            raise SystemExit(gt_proc.stderr or gt_proc.stdout or "ground-truth worker failed")
        dataset.metadata.ground_truth_source = "flatindex-exact-subprocess"
        dataset.metadata.ground_truth_artifact = str(ground_truth_artifact)
        dataset.metadata.ground_truth_k = max(100, args.k)
    dump_json(results_dir / "environment.json", asdict(environment))
    dump_json(results_dir / "dataset.json", asdict(dataset.metadata))

    payload = {
        "config": asdict(config),
        "metric": metric,
        "run_label": run_label,
        "ground_truth_artifact": str(ground_truth_artifact) if ground_truth_artifact else None,
    }
    payload_path = results_dir / "run_payload.json"
    dump_json(payload_path, payload)

    engine_results = []
    for engine_name in engines:
        result_path = results_dir / f"{engine_name}.json"
        proc = run_worker_subprocess(
            engine_name=engine_name,
            worker_payload_path=payload_path,
            result_path=result_path,
        )
        if proc.returncode != 0:
            dump_json(
                result_path,
                failure_result_payload(
                    engine_name=engine_name,
                    metric=metric,
                    requested_k=args.k,
                    evaluation_k=max(100, args.k),
                    run_label=run_label,
                    error=proc.stderr or proc.stdout or "worker failed without output",
                ),
            )
        engine_results.append(json.loads(result_path.read_text(encoding="utf-8")))

    summary = summarize_results(
        config=config,
        dataset=dataset.metadata,
        environment=environment,
        results=[],
    )
    summary["engines"] = sorted(engine_results, key=lambda item: item["engine"])
    dump_json(results_dir / "summary.json", summary)
    print(run_label)
    print(f"results_dir={results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
