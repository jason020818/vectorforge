"""VIBE dataset adapter for ccnews-nomic-768-normalized."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from benchmarks.results import DatasetMetadata

DATASET_NAME = "ccnews-nomic-768-normalized"
HF_REPO_ID = "vector-index-bench/vibe"
HF_FILENAME = f"{DATASET_NAME}.hdf5"
HF_SOURCE_COMMIT = "ca58b61673bb8e9ff5cf2279668161a93ebfbdab"
HF_SOURCE_SHA256 = "c3a498239eea772b7b736f31c25f8dff089f349352ec523b0d9cb694c71f1df3"
EXPECTED_TRAIN_SIZE = 495_328
EXPECTED_DIM = 768
EXPECTED_TEST_SIZE = 1_000
EXPECTED_GT_K = 100


@dataclass(slots=True)
class VibeCcnewsDataset:
    metadata: DatasetMetadata
    vectors: np.ndarray
    queries: np.ndarray
    ground_truth_ids: np.ndarray
    ground_truth_distances: np.ndarray | None


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "h5py is required for VIBE benchmark datasets; install with .[bench]"
        ) from exc
    return h5py


def _hf_download(target_dir: Path) -> Path:
    override = os.environ.get("VECTORFORGE_VIBE_CCNEWS_PATH")
    if override:
        return Path(override).expanduser().resolve()

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "huggingface_hub is required to download the VIBE dataset; install with .[bench]"
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            revision=HF_SOURCE_COMMIT,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        return Path(downloaded).resolve()
    except Exception:
        pass

    direct_path = target_dir / HF_FILENAME
    if direct_path.exists():
        return direct_path.resolve()

    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{HF_SOURCE_COMMIT}/{HF_FILENAME}"
    tmp_path = direct_path.with_suffix(".tmp")
    with urlopen(url) as response, tmp_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp_path.replace(direct_path)
    return direct_path.resolve()


def _require_2d_float32(name: str, array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.dtype != np.float32:
        raise ValueError(f"{name} must preserve float32 representation")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _require_gt(name: str, array: np.ndarray, *, nq: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] != nq:
        raise ValueError(f"{name} query dimension mismatch")
    if array.shape[1] < EXPECTED_GT_K:
        raise ValueError(f"{name} must contain at least {EXPECTED_GT_K} neighbors")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer ids")
    return np.ascontiguousarray(array.astype(np.int64, copy=False))


def _require_distance_array(name: str, array: np.ndarray, *, nq: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] != nq:
        raise ValueError(f"{name} query dimension mismatch")
    if array.shape[1] < EXPECTED_GT_K:
        raise ValueError(f"{name} must contain at least {EXPECTED_GT_K} distances")
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _load_hdf5_arrays(
    path: Path,
    *,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[int, int], int]:
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        required_keys = ("train", "test", "neighbors")
        for key in required_keys:
            if key not in handle:
                raise ValueError(f"malformed VIBE dataset: missing dataset '{key}'")

        train_ds = handle["train"]
        test_ds = handle["test"]
        train_shape = tuple(train_ds.shape)
        test_size = int(test_ds.shape[0])
        vectors = _require_2d_float32(
            "train",
            np.asarray(train_ds[:limit] if limit is not None else train_ds),
        )
        queries = _require_2d_float32("test", np.asarray(test_ds))
        neighbors = _require_gt("neighbors", np.asarray(handle["neighbors"]), nq=queries.shape[0])

        distances = None
        if "distances" in handle:
            distances = _require_distance_array(
                "distances", np.asarray(handle["distances"]), nq=queries.shape[0]
            )

    return vectors, queries, neighbors, distances, train_shape, test_size


def load_vibe_ccnews(
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
    official: bool = False,
) -> VibeCcnewsDataset:
    dataset_dir = cache_dir or Path("benchmarks/datasets/cache")
    path = _hf_download(dataset_dir)

    (
        vectors,
        queries,
        canonical_ids,
        canonical_distances,
        train_shape,
        test_size,
    ) = _load_hdf5_arrays(path, limit=limit)
    if train_shape != (EXPECTED_TRAIN_SIZE, EXPECTED_DIM):
        raise ValueError(
            "unexpected train shape "
            f"{train_shape}, expected {(EXPECTED_TRAIN_SIZE, EXPECTED_DIM)}"
        )
    if queries.shape[1] != EXPECTED_DIM:
        raise ValueError(f"unexpected query dim {queries.shape[1]}, expected {EXPECTED_DIM}")
    if test_size != EXPECTED_TEST_SIZE:
        raise ValueError(f"unexpected query count {test_size}, expected {EXPECTED_TEST_SIZE}")

    is_smoke = limit is not None
    if official and is_smoke:
        raise ValueError(
            "official runs must use the full dataset; --official cannot be combined with --limit"
        )

    ground_truth_source = "vibe-canonical"
    if limit is not None:
        if limit < EXPECTED_GT_K:
            raise ValueError(f"limit must be at least {EXPECTED_GT_K} to compute Recall@100")
        if limit > train_shape[0]:
            raise ValueError(f"limit {limit} exceeds dataset size {train_shape[0]}")
        ground_truth_source = "recompute-required"

    metadata = DatasetMetadata(
        name=DATASET_NAME,
        source=f"hf://datasets/{HF_REPO_ID}/{HF_FILENAME}",
        source_commit=HF_SOURCE_COMMIT,
        source_sha256=HF_SOURCE_SHA256,
        path=str(path),
        split="train/test",
        limit=limit,
        official=official,
        is_smoke=is_smoke,
        n=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
        nq=int(queries.shape[0]),
        dtype="float32",
        metric_hint="cosine",
        ground_truth_source=ground_truth_source,
    )
    return VibeCcnewsDataset(
        metadata=metadata,
        vectors=vectors,
        queries=queries,
        ground_truth_ids=canonical_ids,
        ground_truth_distances=canonical_distances,
    )
