from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest
from vectorforge import FlatIndex


def test_roundtrip_tmp_path(tmp_path: Path) -> None:
    path = tmp_path / "flat.bin"
    db = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    index = FlatIndex(dim=2, metric="l2")
    index.add(db)
    index.save(os.fspath(path))

    loaded = FlatIndex(dim=1, metric="cosine")
    loaded.load(os.fspath(path))
    assert loaded.dim == 2
    assert loaded.metric == "l2"
    assert loaded.ntotal == 2

    q = np.array([[0.0, 0.0]], dtype=np.float32)
    ids_a, dist_a = index.search(q, k=2)
    ids_b, dist_b = loaded.search(q, k=2)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_allclose(dist_a, dist_b, atol=0.0)


def test_load_rejects_invalid_metric(tmp_path: Path) -> None:
    path = tmp_path / "bad_metric.bin"
    path.write_bytes(b"VF01" + struct.pack("<IIIQ", 1, 2, 7, 0))
    index = FlatIndex(dim=2, metric="l2")
    with pytest.raises(RuntimeError):
        index.load(os.fspath(path))


def test_load_rejects_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "truncated_payload.bin"
    payload = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    path.write_bytes(b"VF01" + struct.pack("<IIIQ", 1, 2, 0, 2) + payload)
    index = FlatIndex(dim=2, metric="l2")
    with pytest.raises(RuntimeError):
        index.load(os.fspath(path))


def test_load_rejects_trailing_byte(tmp_path: Path) -> None:
    path = tmp_path / "trailing.bin"
    payload = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    path.write_bytes(b"VF01" + struct.pack("<IIIQ", 1, 2, 0, 1) + payload + b"\x00")
    index = FlatIndex(dim=2, metric="l2")
    index.add(np.array([[0.0, 0.0]], dtype=np.float32))
    with pytest.raises(RuntimeError):
        index.load(os.fspath(path))
    assert index.dim == 2
    assert index.metric == "l2"
    assert index.ntotal == 1
