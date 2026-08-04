from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import (
    CachedDataset,
    Cacher,
    Dataset,
    DatasetConfig,
    DatasetSnapshot,
    DatasetSnapshotStore,
    FileStorageType,
)
from atria_core.datasets._constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR
from atria_core.registry import Registry
from atria_core.transforms import BaseTransform
from atria_core.types import DatasetMetadata, DatasetSplitType, Image, ImageInstance

_synthetic = Registry.group("test_dataset_end_to_end.synthetic")


@_synthetic.register("synthetic")
@pydantic_dataclass(frozen=True)
class SyntheticConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> SyntheticDataset:
        return SyntheticDataset(self, **kwargs)


class _InputTransform:
    def __call__(self, index: int) -> ImageInstance:
        return ImageInstance(
            sample_id=str(index),
            image=Image.from_source(PILImage.new("RGB", (4, 4), color=(index, 0, 0))),
        )


class _RawSplit(Sequence[int]):
    """Indexable raw source -- Dataset wraps this in an
    IndexableSplitIterator automatically, since it's a Sequence.
    isinstance(x, Sequence) checks the ABC registry, not structural
    duck-typing, so this must actually subclass Sequence."""

    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> int:
        if index >= self._count:
            raise IndexError(index)
        return index


class SyntheticDataset(Dataset[SyntheticConfig, ImageInstance]):
    def _download_urls(self) -> list[str]:
        return []

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(description="synthetic test dataset")

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _RawSplit:
        count = 4 if split == DatasetSplitType.train else 2
        return _RawSplit(count)

    def _build_input_transform(self) -> Callable[[Any], ImageInstance]:
        return _InputTransform()


def _record(item: object) -> ImageInstance:
    assert isinstance(item, ImageInstance)
    return item


def _mark_processed(sample: ImageInstance) -> ImageInstance:
    """Module-level (picklable) transform -- multiprocessing.Pool pickles
    task arguments, so closures/lambdas defined inside a test function
    won't survive being sent to a worker process."""
    return replace(sample, sample_id=f"processed-{sample.sample_id}")


class _ConfiguredTransform(BaseTransform):
    prefix: str
    labels: tuple[str, ...] = ("first", "second")

    def __call__(self, sample: ImageInstance) -> ImageInstance:
        return replace(sample, sample_id=f"{self.prefix}-{sample.sample_id}")


def test_build_module_constructs_dataset(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))
    assert isinstance(dataset, SyntheticDataset)
    assert dataset.data_dir == tmp_path

    snapshot = DatasetSnapshot.load(tmp_path)
    assert snapshot.snapshot_kind == "source"
    assert snapshot.dataset_stage == "raw"
    assert snapshot.storage_type is None
    assert snapshot.data_model is None
    assert snapshot.config == dataset.config.to_dict()


def test_dataset_config_from_registry_constructs_typed_config() -> None:
    config = DatasetConfig.from_registry("synthetic")

    assert isinstance(config, SyntheticConfig)
    assert config.dataset_dir_name == "synthetic"


def test_build_module_uses_config_dataset_dir_name(tmp_path: Path) -> None:
    dataset_dir_name = tmp_path.name
    dataset = SyntheticConfig(dataset_dir_name=dataset_dir_name).build_module()

    assert isinstance(dataset, SyntheticDataset)
    assert dataset.data_dir == _DEFAULT_ATRIA_DATASETS_CACHE_DIR / dataset_dir_name


def test_build_module_produces_live_split_iterators(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    train = dataset.split_iterator(DatasetSplitType.train)
    test = dataset.split_iterator(DatasetSplitType.test)

    assert len(train) == 4
    assert len(test) == 2
    assert _record(train[0]).sample_id == "0"


def test_dataset_config_limits_indexable_splits_on_access(tmp_path: Path) -> None:
    dataset = SyntheticConfig(max_train_samples=2, max_test_samples=1).build_module(
        data_dir=str(tmp_path)
    )

    assert len(dataset.train) == 2
    assert len(dataset.test) == 1
    assert [_record(sample).sample_id for sample in dataset.train] == ["0", "1"]
    assert len(dataset._split_iterators[DatasetSplitType.train]) == 4

    snapshot = DatasetSnapshot.load(tmp_path)
    assert snapshot.splits == {"train": 2, "test": 1}


def test_cache_only_writes_configured_max_samples(tmp_path: Path) -> None:
    dataset = SyntheticConfig(max_train_samples=2, max_test_samples=1).build_module(
        data_dir=str(tmp_path)
    )

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).cache(
        dataset, data_dir=str(tmp_path)
    )

    assert len(cached.train) == 2
    assert len(cached.test) == 1
    assert cached.snapshot.splits == {"train": 2, "test": 1}


def test_cache_then_iterate_msgpack(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).cache(
        dataset, data_dir=str(tmp_path)
    )

    train = cached.split_iterator(DatasetSplitType.train)
    test = cached.split_iterator(DatasetSplitType.test)

    assert len(train) == 4
    assert len(test) == 2
    assert {_record(sample).sample_id for sample in train} == {"0", "1", "2", "3"}


def test_cache_is_reused_on_second_call(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))
    cacher = Cacher(FileStorageType.MSGPACK, num_processes=1)
    first = cacher.cache(dataset, data_dir=str(tmp_path))

    dataset2 = SyntheticConfig().build_module(data_dir=str(tmp_path))
    second = cacher.cache(dataset2, data_dir=str(tmp_path))

    assert first.data_dir == second.data_dir


def test_process_and_cache_applies_transform_at_write_time(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).process_and_cache(
        dataset, _mark_processed, data_dir=str(tmp_path)
    )

    train = cached.split_iterator(DatasetSplitType.train)
    assert {_record(sample).sample_id for sample in train} == {
        "processed-0",
        "processed-1",
        "processed-2",
        "processed-3",
    }
    assert cached.dataset_stage == "processed"


def test_cached_snapshot_stores_dataclass_transform_params(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).process_and_cache(
        dataset, _ConfiguredTransform(prefix="configured"), data_dir=str(tmp_path)
    )

    assert cached.snapshot.transforms == [
        {
            "type": "atria_core.datasets._cacher.PreprocessTransform",
            "params": {
                "materialize_content": True,
                "resize_images": False,
                "image_max_size": None,
            },
        },
        {
            "type": f"{_ConfiguredTransform.__module__}.{_ConfiguredTransform.__qualname__}",
            "params": {
                "prefix": "configured",
                "labels": ["first", "second"],
            },
        },
    ]


def test_cache_with_multiprocessing_num_processes_gt_1(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=2).cache(
        dataset, data_dir=str(tmp_path)
    )

    train = cached.split_iterator(DatasetSplitType.train)
    test = cached.split_iterator(DatasetSplitType.test)
    assert len(train) == 4
    assert len(test) == 2
    assert {_record(sample).sample_id for sample in train} == {"0", "1", "2", "3"}


def test_cacher_validate_cache_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    assert Cacher.validate_cache(tmp_path) is False

    (tmp_path / "snapshot.yaml").write_text("dataset_class_name: Foo\n")
    assert Cacher.validate_cache(tmp_path) is False


def test_cache_writes_discoverable_versioned_snapshot(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))
    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).cache(
        dataset, data_dir=str(tmp_path)
    )

    root_snapshot = DatasetSnapshot.load(tmp_path)
    assert root_snapshot.snapshot_kind == "source"

    snapshot = DatasetSnapshot.load(cached.data_dir)
    assert snapshot.schema_version == DatasetSnapshot.CURRENT_SCHEMA_VERSION
    assert snapshot.created_at is not None
    assert snapshot.snapshot_kind == "cached"
    assert snapshot.splits == {"train": 4, "test": 2}
    assert snapshot.dataset_stage == "raw"
    assert snapshot.config == dataset.config.to_dict()
    assert snapshot.metadata == dataset.metadata.to_dict()
    assert not (cached.data_dir / "snapshot.yaml.tmp").exists()
    assert not (cached.data_dir / "config.yaml").exists()
    assert not (cached.data_dir / "metadata.yaml").exists()

    discovered = DatasetSnapshot.discover(tmp_path)
    assert [item.path for item in discovered] == [tmp_path, cached.data_dir]
    source_discovered = DatasetSnapshotStore.discover(tmp_path, snapshot_kind="source")
    assert [item.path for item in source_discovered] == [tmp_path]
    cache_discovered = DatasetSnapshotStore.discover(tmp_path, snapshot_kind="cached")
    assert [item.path for item in cache_discovered] == [cached.data_dir]
    reopened = [CachedDataset(item.path) for item in cache_discovered if item.path]
    assert [item.data_dir for item in reopened] == [cached.data_dir]


def test_snapshot_loads_legacy_schema(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("_target_: example.Config\n")
    (tmp_path / "metadata.yaml").write_text("{}\n")
    (tmp_path / "snapshot.yaml").write_text(
        "storage_type: msgpack\n"
        "data_model: example.Model\n"
        "dataset_class_name: Example\n"
        "config_name: Example-123\n"
        "config_hash: '123'\n"
    )

    snapshot = DatasetSnapshot.load(tmp_path)
    assert snapshot.schema_version == 0
    assert snapshot.created_at is None
    assert snapshot.splits == {}
    assert snapshot.snapshot_kind == "cached"
    assert snapshot.dataset_stage == "raw"
    assert snapshot.config == {}
    assert snapshot.metadata == {}
    assert DatasetSnapshot.validate(tmp_path)
