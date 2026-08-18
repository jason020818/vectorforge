from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.engines.base import package_version
from benchmarks.environment import EnvironmentMetadata, ensure_official_mode_allowed
from benchmarks.phase3 import failure_result_payload
from benchmarks.phase3 import main as phase3_main
from benchmarks.runner import benchmark_engine, run_worker_subprocess


def test_official_mode_rejects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "phase3.py",
            "--engine",
            "vectorforge",
            "--official",
            "--limit",
            "10000",
        ],
    )
    with pytest.raises(SystemExit, match="--official cannot be combined with --limit"):
        phase3_main()


def test_failure_result_payload_is_structured() -> None:
    payload = failure_result_payload(
        engine_name="faiss",
        metric="cosine",
        requested_k=10,
        evaluation_k=100,
        run_label="NON-OFFICIAL SMOKE RESULT",
        error="boom",
    )
    assert payload["engine"] == "faiss"
    assert payload["available"] is False
    assert payload["error"] == "boom"


def test_official_environment_cannot_be_bypassed() -> None:
    env = EnvironmentMetadata(
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
        thp_state="always",
        official_environment_ready=False,
        warnings=["wsl"],
        env_threads={"OMP_NUM_THREADS": "1"},
    )
    with pytest.raises(ValueError, match="may only produce NON-OFFICIAL results"):
        ensure_official_mode_allowed(env, official=True, allow_uncontrolled_environment=True)


class _UnavailableAdapter:
    name = "fake"

    @staticmethod
    def dependency_available() -> bool:
        return False


def test_missing_optional_dependency_returns_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmarks.runner.create_engine", lambda _name: _UnavailableAdapter())
    result = benchmark_engine(
        engine_name="fake",
        vectors=__import__("numpy").zeros((10, 4), dtype=__import__("numpy").float32),
        queries=__import__("numpy").zeros((2, 4), dtype=__import__("numpy").float32),
        gt_ids=__import__("numpy").zeros((2, 100), dtype=__import__("numpy").int64),
        metric="cosine",
        requested_k=10,
        evaluation_k=100,
        M=16,
        ef_construction=200,
        ef_search=100,
        warmup=1,
        repeat=1,
        threads=1,
        run_label="NON-OFFICIAL SMOKE RESULT",
    )
    assert result.available is False
    assert result.dependency_available is False
    assert "optional benchmark dependency" in (result.error or "")


def test_worker_environment_is_set_before_subprocess_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs["env"]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("benchmarks.runner.subprocess.run", fake_run)
    run_worker_subprocess(
        engine_name="vectorforge",
        worker_payload_path=tmp_path / "payload.json",
        result_path=tmp_path / "result.json",
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert captured["env"][name] == "1"


def test_package_version_reporting_uses_metadata() -> None:
    assert package_version("hnswlib") != "unknown"
