"""Common result schema and deterministic JSON helpers for Phase 3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    if pct < 0.0 or pct > 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def recall_at_k(ann_ids: list[list[int]], gt_ids: list[list[int]], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ann_ids) != len(gt_ids):
        raise ValueError("ann_ids and gt_ids must have the same number of queries")
    if not ann_ids:
        return 0.0

    hits = 0
    for ann_row, gt_row in zip(ann_ids, gt_ids):
        if len(ann_row) < k or len(gt_row) < k:
            raise ValueError("each ANN and ground-truth row must contain at least k ids")
        hits += len(set(ann_row[:k]).intersection(set(gt_row[:k])))
    return hits / float(len(ann_ids) * k)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(slots=True)
class DatasetMetadata:
    name: str
    source: str
    source_commit: str | None
    source_sha256: str | None
    verified_sha256: str | None
    path: str
    split: str
    limit: int | None
    official: bool
    is_smoke: bool
    n: int
    dim: int
    nq: int
    dtype: str
    metric_hint: str
    ground_truth_source: str
    ground_truth_artifact: str | None
    ground_truth_k: int


@dataclass(slots=True)
class EnvironmentMetadata:
    cpu_model: str
    logical_cpus: int | None
    physical_cpus: int | None
    cpu_affinity: list[int] | None
    os: str
    kernel: str
    python_version: str
    total_ram_bytes: int | None
    is_wsl: bool
    smt_status: str | None
    cpu_governor: str | None
    thp_state: str | None
    official_environment_ready: bool
    warnings: list[str]
    env_threads: dict[str, str | None]


@dataclass(slots=True)
class RunConfig:
    dataset: str
    engine: str
    metric: str
    k: int
    evaluation_k: int
    M: int
    ef_construction: int
    ef_search: int
    warmup: int
    repeat: int
    limit: int | None
    official: bool
    requested_threads: int
    allow_uncontrolled_environment: bool
    git_commit: str
    timestamp_utc: str
    results_dir: str


@dataclass(slots=True)
class RepeatMetrics:
    elapsed_s: float
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: int


@dataclass(slots=True)
class EngineResult:
    engine: str
    available: bool
    dependency_available: bool
    error: str | None
    metric: str
    requested_k: int
    evaluation_k: int
    build_time_s: float | None
    recall_at_10: float | None
    recall_at_100: float | None
    qps_median: float | None
    p50_ms_median: float | None
    p95_ms_median: float | None
    p99_ms_median: float | None
    repeats: list[RepeatMetrics]
    peak_rss_bytes: int | None
    baseline_rss_bytes: int | None
    post_build_rss_bytes: int | None
    index_size_bytes: int | None
    version: dict[str, str]
    parameters: dict[str, Any]
    worker_thread_env: dict[str, str | None]
    effective_threads: str
    memory_scope: str
    run_label: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def summarize_results(
    config: RunConfig,
    dataset: DatasetMetadata,
    environment: EnvironmentMetadata,
    results: list[EngineResult],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.engine):
        summary_rows.append(
            {
                "engine": result.engine,
                "available": result.available,
                "dependency_available": result.dependency_available,
                "error": result.error,
                "recall_at_10": result.recall_at_10,
                "recall_at_100": result.recall_at_100,
                "qps_median": result.qps_median,
                "p50_ms_median": result.p50_ms_median,
                "p95_ms_median": result.p95_ms_median,
                "p99_ms_median": result.p99_ms_median,
                "build_time_s": result.build_time_s,
                "peak_rss_bytes": result.peak_rss_bytes,
                "index_size_bytes": result.index_size_bytes,
                "run_label": result.run_label,
            }
        )

    return {
        "config": asdict(config),
        "dataset": asdict(dataset),
        "environment": asdict(environment),
        "engines": summary_rows,
        "main_statistics": {
            "qps_statistic": "median",
            "latency_statistic": "median_of_repeat_percentiles",
            "repeat_count": config.repeat,
            "warmup_count": config.warmup,
        },
    }


def finalize_repeat_metrics(latency_samples_ms: list[float], elapsed_s: float) -> RepeatMetrics:
    if elapsed_s <= 0.0:
        raise ValueError("elapsed_s must be positive")
    return RepeatMetrics(
        elapsed_s=elapsed_s,
        qps=len(latency_samples_ms) / elapsed_s,
        p50_ms=percentile(latency_samples_ms, 50.0),
        p95_ms=percentile(latency_samples_ms, 95.0),
        p99_ms=percentile(latency_samples_ms, 99.0),
        samples=len(latency_samples_ms),
    )


def median_repeat_value(repeats: list[RepeatMetrics], field: str) -> float | None:
    if not repeats:
        return None
    return median([float(getattr(item, field)) for item in repeats])
