from __future__ import annotations

import pytest

from benchmarks.phase3 import failure_result_payload
from benchmarks.phase3 import main as phase3_main
from benchmarks.runner import benchmark_engine


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
