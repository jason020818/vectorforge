from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.datasets import vibe_ccnews

h5py = pytest.importorskip("h5py")


def _write_dataset(path: Path, *, bad_value: float | None = None) -> None:
    train = np.ones((128, 4), dtype=np.float32)
    test = np.ones((3, 4), dtype=np.float32)
    if bad_value is not None:
        train[0, 0] = bad_value
    neighbors = np.tile(np.arange(100, dtype=np.int64), (3, 1))
    distances = np.zeros((3, 100), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=train)
        handle.create_dataset("test", data=test)
        handle.create_dataset("neighbors", data=neighbors)
        handle.create_dataset("distances", data=distances)


def _set_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TRAIN_SIZE", 128)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TEST_SIZE", 3)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_DIM", 4)


def _set_expected_sha(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        vibe_ccnews,
        "HF_SOURCE_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _write_sha_cache(path: Path, sha256: str, *, stat: os.stat_result | None = None) -> None:
    file_stat = stat or path.stat()
    vibe_ccnews._sha256_cache_path(path).write_text(
        json.dumps(
            {
                "path": str(path),
                "size": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
                "sha256": sha256,
            }
        ),
        encoding="utf-8",
    )


def test_vibe_loader_rejects_malformed_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "malformed.hdf5"
    train = np.ones((128, 4), dtype=np.float32)
    test = np.ones((3, 4), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=train)
        handle.create_dataset("test", data=test)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    _set_expected_sha(monkeypatch, path)
    with pytest.raises(ValueError, match="missing dataset 'neighbors'"):
        vibe_ccnews.load_vibe_ccnews()


def test_vibe_loader_rejects_non_finite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "bad.hdf5"
    _write_dataset(path, bad_value=np.nan)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    _set_expected_sha(monkeypatch, path)
    with pytest.raises(ValueError, match="non-finite"):
        vibe_ccnews.load_vibe_ccnews()


def test_vibe_loader_smoke_limit_marks_non_official(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "ok.hdf5"
    _write_dataset(path)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    _set_expected_sha(monkeypatch, path)
    dataset = vibe_ccnews.load_vibe_ccnews(limit=100, official=False)
    assert dataset.metadata.is_smoke is True
    assert dataset.metadata.official is False
    assert dataset.metadata.ground_truth_source == "recompute-required"
    assert dataset.metadata.verified_sha256 == vibe_ccnews.HF_SOURCE_SHA256


def test_vibe_loader_rejects_official_with_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "ok.hdf5"
    _write_dataset(path)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    _set_expected_sha(monkeypatch, path)
    with pytest.raises(ValueError, match="official runs must use the full dataset"):
        vibe_ccnews.load_vibe_ccnews(limit=100, official=True)


def test_vibe_loader_verifies_sha256_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "sha_ok.hdf5"
    _write_dataset(path)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(vibe_ccnews, "HF_SOURCE_SHA256", expected)
    dataset = vibe_ccnews.load_vibe_ccnews()
    assert dataset.metadata.verified_sha256 == expected


def test_vibe_loader_rejects_sha256_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "sha_bad.hdf5"
    _write_dataset(path)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    monkeypatch.setattr(vibe_ccnews, "HF_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="dataset SHA256 mismatch"):
        vibe_ccnews.load_vibe_ccnews()


def test_vibe_loader_smoke_can_reuse_valid_sha_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "cached.hdf5"
    _write_dataset(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_sha_cache(path, expected)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    monkeypatch.setattr(vibe_ccnews, "HF_SOURCE_SHA256", expected)
    monkeypatch.setattr(
        vibe_ccnews,
        "_compute_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("smoke should reuse valid cache")),
    )
    dataset = vibe_ccnews.load_vibe_ccnews(official=False)
    assert dataset.metadata.verified_sha256 == expected


def test_vibe_loader_official_ignores_valid_cache_and_recomputes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "official.hdf5"
    _write_dataset(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_sha_cache(path, expected)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    monkeypatch.setattr(vibe_ccnews, "HF_SOURCE_SHA256", expected)
    calls = {"count": 0}

    def fake_compute(_path: Path) -> str:
        calls["count"] += 1
        return expected

    monkeypatch.setattr(vibe_ccnews, "_compute_sha256", fake_compute)
    dataset = vibe_ccnews.load_vibe_ccnews(official=True)
    assert calls["count"] == 1
    assert dataset.metadata.verified_sha256 == expected


def test_vibe_loader_official_detects_changed_content_despite_forged_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "forged.hdf5"
    _write_dataset(path)
    original_stat = path.stat()
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_sha_cache(path, expected, stat=original_stat)

    altered = np.full((128, 4), 2.0, dtype=np.float32)
    test = np.ones((3, 4), dtype=np.float32)
    neighbors = np.tile(np.arange(100, dtype=np.int64), (3, 1))
    distances = np.zeros((3, 100), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=altered)
        handle.create_dataset("test", data=test)
        handle.create_dataset("neighbors", data=neighbors)
        handle.create_dataset("distances", data=distances)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    _write_sha_cache(path, expected, stat=original_stat)

    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    _set_expected(monkeypatch)
    monkeypatch.setattr(vibe_ccnews, "HF_SOURCE_SHA256", expected)
    with pytest.raises(ValueError, match="dataset SHA256 mismatch"):
        vibe_ccnews.load_vibe_ccnews(official=True)
