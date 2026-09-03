from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from atria_core.datasets import CachedDataset, DatasetBuilder, FileStorageType
from atria_core.types import DatasetSplitType, ImageInstance

# Registering the synthetic dataset is a side effect of importing this module.
from tests.datasets.test_dataset_end_to_end import SyntheticDataset  # noqa: F401


def _mark_processed(sample: ImageInstance) -> ImageInstance:
    """Module-level (picklable) transform -- see test_dataset_end_to_end.py."""
    return replace(sample, sample_id=f"processed-{sample.sample_id}")


def _mark_other(sample: ImageInstance) -> ImageInstance:
    return replace(sample, sample_id=f"other-{sample.sample_id}")


def test_load_without_cache_returns_the_source_dataset(tmp_path: Path) -> None:
    dataset = DatasetBuilder().load("synthetic", data_dir=str(tmp_path)).build()

    assert isinstance(dataset, SyntheticDataset)
    assert not (tmp_path / "artifacts").exists()


def test_load_then_cache_returns_a_cached_dataset(tmp_path: Path) -> None:
    dataset = (
        DatasetBuilder()
        .load("synthetic", data_dir=str(tmp_path))
        .cache(FileStorageType.MSGPACK, num_processes=1)
        .build()
    )

    assert isinstance(dataset, CachedDataset)
    assert len(dataset.train) == 4
    assert len(dataset.test) == 2


def test_process_and_cache_bakes_the_transform_into_the_cache(tmp_path: Path) -> None:
    dataset = (
        DatasetBuilder()
        .load("synthetic", data_dir=str(tmp_path))
        .process_and_cache(FileStorageType.MSGPACK, _mark_processed, num_processes=1)
        .build()
    )
    assert isinstance(dataset, CachedDataset)

    reopened = CachedDataset(dataset.data_dir)

    assert {sample.sample_id for sample in reopened.train} == {
        "processed-0",
        "processed-1",
        "processed-2",
        "processed-3",
    }


def test_different_transforms_produce_different_cache_directories(
    tmp_path: Path,
) -> None:
    first = (
        DatasetBuilder()
        .load("synthetic", data_dir=str(tmp_path))
        .process_and_cache(FileStorageType.MSGPACK, _mark_processed, num_processes=1)
        .build()
    )
    second = (
        DatasetBuilder()
        .load("synthetic", data_dir=str(tmp_path))
        .process_and_cache(FileStorageType.MSGPACK, _mark_other, num_processes=1)
        .build()
    )

    assert first.data_dir != second.data_dir


def test_cache_before_load_is_rejected() -> None:
    with pytest.raises(ValueError, match="call load\\(\\) before cache\\(\\)"):
        DatasetBuilder().cache(FileStorageType.MSGPACK)


def test_process_and_cache_before_load_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="call load\\(\\) before process_and_cache\\(\\)"
    ):
        DatasetBuilder().process_and_cache(FileStorageType.MSGPACK, _mark_processed)


def test_build_before_load_is_rejected() -> None:
    with pytest.raises(ValueError, match="call load\\(\\) before build\\(\\)"):
        DatasetBuilder().build()


def test_load_params_reach_the_config(tmp_path: Path) -> None:
    dataset = (
        DatasetBuilder()
        .load("synthetic", max_train_samples=2, data_dir=str(tmp_path))
        .build()
    )

    assert dataset.config.max_train_samples == 2
    assert len(dataset.train) == 2


def test_unknown_load_param_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonsense"):
        DatasetBuilder().load("synthetic", nonsense=1, data_dir=str(tmp_path))


def test_split_builds_only_that_split(tmp_path: Path) -> None:
    dataset = (
        DatasetBuilder()
        .load("synthetic", split=DatasetSplitType.train, data_dir=str(tmp_path))
        .build()
    )

    assert list(dataset.split_iterators) == [DatasetSplitType.train]


def test_data_dir_never_appears_in_config(tmp_path: Path) -> None:
    dataset = DatasetBuilder().load("synthetic", data_dir=str(tmp_path)).build()

    assert "data_dir" not in dataset.config.to_dict()
