from __future__ import annotations

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
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TRAIN_SIZE", 128)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TEST_SIZE", 3)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_DIM", 4)
    with pytest.raises(ValueError, match="missing dataset 'neighbors'"):
        vibe_ccnews.load_vibe_ccnews()


def test_vibe_loader_rejects_non_finite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "bad.hdf5"
    _write_dataset(path, bad_value=np.nan)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TRAIN_SIZE", 128)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TEST_SIZE", 3)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_DIM", 4)
    with pytest.raises(ValueError, match="non-finite"):
        vibe_ccnews.load_vibe_ccnews()


def test_vibe_loader_smoke_limit_marks_non_official(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "ok.hdf5"
    _write_dataset(path)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TRAIN_SIZE", 128)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TEST_SIZE", 3)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_DIM", 4)
    dataset = vibe_ccnews.load_vibe_ccnews(limit=100, official=False)
    assert dataset.metadata.is_smoke is True
    assert dataset.metadata.official is False
    assert dataset.metadata.ground_truth_source == "recompute-required"


def test_vibe_loader_rejects_official_with_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "ok.hdf5"
    _write_dataset(path)
    monkeypatch.setenv("VECTORFORGE_VIBE_CCNEWS_PATH", str(path))
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TRAIN_SIZE", 128)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_TEST_SIZE", 3)
    monkeypatch.setattr(vibe_ccnews, "EXPECTED_DIM", 4)
    with pytest.raises(ValueError, match="official runs must use the full dataset"):
        vibe_ccnews.load_vibe_ccnews(limit=100, official=True)
