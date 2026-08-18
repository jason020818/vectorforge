from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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
