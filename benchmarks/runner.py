"""Shared benchmark runner helpers."""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from vectorforge import FlatIndex

from benchmarks.datasets import load_vibe_ccnews
from benchmarks.engines import BenchmarkAdapterConfig, create_engine
from benchmarks.environment import (
    apply_single_thread_env,
    single_thread_env,
    worker_thread_env_snapshot,
)
from benchmarks.results import (
    EngineResult,
    finalize_repeat_metrics,
    median_repeat_value,
    recall_at_k,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency path
    psutil = None


def load_dataset_for_run(name: str, *, limit: int | None, official: bool):
    if name != "ccnews-nomic-768-normalized":
        raise ValueError(f"unsupported dataset '{name}'")
    return load_vibe_ccnews(limit=limit, official=official)


def effective_metric(dataset_name: str, requested_metric: str) -> str:
    if dataset_name.endswith("-normalized") and requested_metric not in {"cosine", "l2"}:
        raise ValueError("normalized dataset only supports l2/cosine in this harness")
    return requested_metric


def exact_ground_truth(vectors: np.ndarray, queries: np.ndarray, metric: str, k: int) -> np.ndarray:
    index = FlatIndex(dim=int(vectors.shape[1]), metric=metric)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    ids, _ = index.search(np.ascontiguousarray(queries, dtype=np.float32), k=k)
    return np.asarray(ids, dtype=np.int64)


def _require_psutil():
    if psutil is None:
        raise RuntimeError(
            "psutil is required for benchmark RSS measurement; install with .[bench]"
        )
    return psutil


def benchmark_engine(
    *,
    engine_name: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    gt_ids: np.ndarray,
    metric: str,
    requested_k: int,
    evaluation_k: int,
    M: int,
    ef_construction: int,
    ef_search: int,
    warmup: int,
    repeat: int,
    threads: int,
    run_label: str,
) -> EngineResult:
    apply_single_thread_env()
    adapter = create_engine(engine_name)
    if not adapter.dependency_available():
        return EngineResult(
            engine=engine_name,
            available=False,
            dependency_available=False,
            error="optional benchmark dependency is not installed",
            metric=metric,
            requested_k=requested_k,
            evaluation_k=evaluation_k,
            build_time_s=None,
            recall_at_10=None,
            recall_at_100=None,
            qps_median=None,
            p50_ms_median=None,
            p95_ms_median=None,
            p99_ms_median=None,
            repeats=[],
            peak_rss_bytes=None,
            baseline_rss_bytes=None,
            post_build_rss_bytes=None,
            index_size_bytes=None,
            version={},
            parameters={},
            worker_thread_env=worker_thread_env_snapshot(),
            effective_threads="not_available",
            memory_scope="isolated engine worker process only; excludes ground-truth subprocess",
            run_label=run_label,
        )

    config = BenchmarkAdapterConfig(
        dim=int(vectors.shape[1]),
        metric=metric,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        threads=threads,
    )
    proc = _require_psutil().Process()
    baseline_rss = proc.memory_info().rss
    t0 = time.perf_counter()
    adapter.build(vectors, config)
    build_time_s = time.perf_counter() - t0
    post_build_rss = proc.memory_info().rss

    for _ in range(warmup):
        adapter.search(queries, requested_k)

    repeats = []
    for _ in range(repeat):
        per_query_ms: list[float] = []
        start = time.perf_counter()
        for row in queries:
            q0 = time.perf_counter()
            adapter.search(np.ascontiguousarray(row[None, :], dtype=np.float32), requested_k)
            per_query_ms.append((time.perf_counter() - q0) * 1000.0)
        elapsed = time.perf_counter() - start
        repeats.append(finalize_repeat_metrics(per_query_ms, elapsed))

    ids_eval, _ = adapter.search(queries, evaluation_k)
    ann_ids = np.asarray(ids_eval, dtype=np.int64)
    gt_eval = np.asarray(gt_ids[:, :evaluation_k], dtype=np.int64)

    with tempfile.TemporaryDirectory(prefix=f"vectorforge-{engine_name}-") as temp_dir:
        out_path = Path(temp_dir) / f"{engine_name}.index"
        try:
            adapter.save(out_path)
            index_size = adapter.index_size_bytes(out_path)
        except Exception:
            index_size = None

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        peak_rss *= 1024

    return EngineResult(
        engine=engine_name,
        available=True,
        dependency_available=True,
        error=None,
        metric=metric,
        requested_k=requested_k,
        evaluation_k=evaluation_k,
        build_time_s=build_time_s,
        recall_at_10=recall_at_k(ann_ids.tolist(), gt_eval.tolist(), 10),
        recall_at_100=recall_at_k(ann_ids.tolist(), gt_eval.tolist(), 100),
        qps_median=median_repeat_value(repeats, "qps"),
        p50_ms_median=median_repeat_value(repeats, "p50_ms"),
        p95_ms_median=median_repeat_value(repeats, "p95_ms"),
        p99_ms_median=median_repeat_value(repeats, "p99_ms"),
        repeats=repeats,
        peak_rss_bytes=int(peak_rss),
        baseline_rss_bytes=baseline_rss,
        post_build_rss_bytes=post_build_rss,
        index_size_bytes=index_size,
        version=adapter.version_info(),
        parameters=adapter.actual_parameters(),
        worker_thread_env=worker_thread_env_snapshot(),
        effective_threads=adapter.effective_threads(),
        memory_scope="isolated engine worker process only; excludes ground-truth subprocess",
        run_label=run_label,
    )


def run_worker_subprocess(
    *,
    engine_name: str,
    worker_payload_path: Path,
    result_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(single_thread_env())
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.worker",
            "--payload",
            str(worker_payload_path),
            "--result",
            str(result_path),
            "--engine",
            engine_name,
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def run_ground_truth_subprocess(
    *,
    payload_path: Path,
    result_path: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(single_thread_env())
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ground_truth_worker",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
