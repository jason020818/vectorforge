from __future__ import annotations

from benchmarks.results import (
    DatasetMetadata,
    EngineResult,
    EnvironmentMetadata,
    RepeatMetrics,
    RunConfig,
    percentile,
    recall_at_k,
    summarize_results,
)


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50.0) == 2.5
    assert percentile(values, 95.0) > 3.0


def test_recall_at_k_matches_definition() -> None:
    ann = [[0, 2, 3], [3, 4, 5]]
    gt = [[0, 1, 2], [3, 5, 6]]
    assert recall_at_k(ann, gt, 3) == 4.0 / 6.0


def test_summary_generation_is_deterministic() -> None:
    config = RunConfig(
        dataset="ccnews-nomic-768-normalized",
        engine="all",
        metric="cosine",
        k=10,
        evaluation_k=100,
        M=16,
        ef_construction=200,
        ef_search=100,
        warmup=2,
        repeat=2,
        limit=10000,
        official=False,
        requested_threads=1,
        allow_uncontrolled_environment=False,
        git_commit="abc123",
        timestamp_utc="20260818T000000Z",
        results_dir="benchmarks/results/x",
    )
    dataset = DatasetMetadata(
        name="ccnews-nomic-768-normalized",
        source="hf://datasets/vector-index-bench/vibe/ccnews-nomic-768-normalized.hdf5",
        source_commit="c",
        source_sha256="s",
        path="x",
        split="train/test",
        limit=10000,
        official=False,
        is_smoke=True,
        n=10000,
        dim=768,
        nq=1000,
        dtype="float32",
        metric_hint="cosine",
        ground_truth_source="recompute-required",
    )
    environment = EnvironmentMetadata(
        cpu_model="cpu",
        logical_cpus=8,
        physical_cpus=4,
        cpu_affinity=[0],
        os="Linux",
        kernel="6.x",
        python_version="3.12",
        total_ram_bytes=1,
        is_wsl=True,
        smt_status="on",
        cpu_governor="powersave",
        thp_state="always [madvise] never",
        official_environment_ready=False,
        warnings=["wsl"],
        env_threads={"OMP_NUM_THREADS": "1"},
    )
    result = EngineResult(
        engine="vectorforge",
        available=True,
        dependency_available=True,
        error=None,
        metric="cosine",
        requested_k=10,
        evaluation_k=100,
        build_time_s=1.0,
        recall_at_10=0.95,
        recall_at_100=0.80,
        qps_median=1000.0,
        p50_ms_median=0.1,
        p95_ms_median=0.2,
        p99_ms_median=0.3,
        repeats=[RepeatMetrics(1.0, 1000.0, 0.1, 0.2, 0.3, 10)],
        peak_rss_bytes=10,
        baseline_rss_bytes=5,
        post_build_rss_bytes=7,
        index_size_bytes=100,
        version={"vectorforge": "0.1.0"},
        parameters={"M": 16},
        run_label="NON-OFFICIAL SMOKE RESULT",
    )
    a = summarize_results(config, dataset, environment, [result])
    b = summarize_results(config, dataset, environment, [result])
    assert a == b
