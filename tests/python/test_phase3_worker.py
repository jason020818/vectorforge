from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.results import DatasetMetadata, EngineResult
from benchmarks.worker import main as worker_main


def test_worker_uses_shared_ground_truth_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gt_path = tmp_path / "ground_truth.npy"
    np.save(gt_path, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": "ccnews-nomic-768-normalized",
                    "engine": "vectorforge",
                    "metric": "cosine",
                    "k": 10,
                    "evaluation_k": 100,
                    "M": 16,
                    "ef_construction": 200,
                    "ef_search": 100,
                    "warmup": 1,
                    "repeat": 1,
                    "limit": 100,
                    "official": False,
                    "requested_threads": 1,
                    "allow_uncontrolled_environment": False,
                    "git_commit": "abc",
                    "timestamp_utc": "now",
                    "results_dir": "x",
                },
                "metric": "cosine",
                "run_label": "NON-OFFICIAL SMOKE RESULT",
                "ground_truth_artifact": str(gt_path),
            }
        ),
        encoding="utf-8",
    )

    dataset = type(
        "Dataset",
        (),
        {
            "metadata": DatasetMetadata(
                name="ccnews-nomic-768-normalized",
                source="hf://x",
                source_commit="c",
                source_sha256="s",
                verified_sha256="s",
                path="x",
                split="train/test",
                limit=100,
                official=False,
                is_smoke=True,
                n=100,
                dim=4,
                nq=2,
                dtype="float32",
                metric_hint="cosine",
                ground_truth_source="flatindex-exact-subprocess",
                ground_truth_artifact=str(gt_path),
                ground_truth_k=100,
            ),
            "vectors": np.zeros((100, 4), dtype=np.float32),
            "queries": np.zeros((2, 4), dtype=np.float32),
            "ground_truth_ids": np.zeros((2, 100), dtype=np.int64),
            "ground_truth_distances": None,
        },
    )()

    seen = {}

    def fake_benchmark_engine(**kwargs):
        seen["gt_ids"] = kwargs["gt_ids"]
        return EngineResult(
            engine="vectorforge",
            available=True,
            dependency_available=True,
            error=None,
            metric="cosine",
            requested_k=10,
            evaluation_k=100,
            build_time_s=1.0,
            recall_at_10=1.0,
            recall_at_100=1.0,
            qps_median=1.0,
            p50_ms_median=1.0,
            p95_ms_median=1.0,
            p99_ms_median=1.0,
            repeats=[],
            peak_rss_bytes=1,
            baseline_rss_bytes=1,
            post_build_rss_bytes=1,
            index_size_bytes=1,
            version={"vectorforge": "0.1.0"},
            parameters={},
            worker_thread_env={"OMP_NUM_THREADS": "1"},
            effective_threads="1",
            memory_scope="isolated engine worker process only; excludes ground-truth subprocess",
            run_label="NON-OFFICIAL SMOKE RESULT",
        )

    monkeypatch.setattr("benchmarks.worker.load_dataset_for_run", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        "benchmarks.worker.exact_ground_truth",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not recompute GT")),
    )
    monkeypatch.setattr("benchmarks.worker.benchmark_engine", fake_benchmark_engine)
    monkeypatch.setattr(
        "sys.argv",
        [
            "worker.py",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--engine",
            "vectorforge",
        ],
    )
    assert worker_main() == 0
    np.testing.assert_array_equal(seen["gt_ids"], np.load(gt_path))
