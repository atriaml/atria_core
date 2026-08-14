from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
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
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.transforms import BaseTransform
from atria_core.types import DatasetMetadata, DatasetSplitType, Image, ImageInstance


@pydantic_dataclass(frozen=True)
class SyntheticConfig(DatasetConfig):
    max_train_samples: int | None = None
    max_test_samples: int | None = None


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


class SyntheticDataset(Dataset[ImageInstance, SyntheticConfig]):
    def _download_urls(self) -> list[UrlSpec]:
        return []

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(description="synthetic test dataset")

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _RawSplit:
        if split == DatasetSplitType.train:
            return _RawSplit(min(4, self.config.max_train_samples or 4))
        return _RawSplit(min(2, self.config.max_test_samples or 2))

    def _build_input_transform(self) -> Callable[[Any], ImageInstance]:
        return _InputTransform()


@pydantic_dataclass(frozen=True)
class EmptyConfig(DatasetConfig):
    pass


class EmptyConfigDataset(Dataset[ImageInstance, EmptyConfig]):
    def _download_urls(self) -> list[UrlSpec]:
        return []

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(description="empty-config synthetic dataset")

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _RawSplit:
        return _RawSplit(1)

    def _build_input_transform(self) -> Callable[[Any], ImageInstance]:
        return _InputTransform()


def synthetic(
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    **kwargs: Any,
) -> SyntheticDataset:
    """Build a SyntheticDataset from plain params.

    This is the whole public entry point -- an importable function, so callers
    get the exact return type with no registry in between.
    """
    return SyntheticDataset(
        config=SyntheticConfig(
            max_train_samples=max_train_samples, max_test_samples=max_test_samples
        ),
        **kwargs,
    )


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


def test_dataset_constructs_with_default_config(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))
    assert isinstance(dataset, SyntheticDataset)
    assert dataset.data_dir == tmp_path

    snapshot = DatasetSnapshot.load(tmp_path)
    assert snapshot.snapshot_kind == "source"
    assert snapshot.dataset_stage == "raw"
    assert snapshot.storage_type is None
    assert snapshot.data_model is None
    assert snapshot.config == dataset.config.to_dict()


def test_default_config_is_used_when_none_given(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))

    assert isinstance(dataset.config, SyntheticConfig)
    assert dataset.config == SyntheticConfig()


def test_mismatched_config_class_is_rejected(tmp_path: Path) -> None:
    @pydantic_dataclass(frozen=True)
    class OtherConfig(DatasetConfig):
        pass

    with pytest.raises(TypeError, match="takes a SyntheticConfig, but got OtherConfig"):
        SyntheticDataset(config=OtherConfig(), data_dir=str(tmp_path))  # type: ignore[arg-type]


def test_factory_builds_dataset_from_params(tmp_path: Path) -> None:
    dataset = synthetic(max_train_samples=2, data_dir=str(tmp_path))

    assert isinstance(dataset, SyntheticDataset)
    assert dataset.config.max_train_samples == 2


def test_factory_is_reachable_by_import_path(tmp_path: Path) -> None:
    """A name arriving as data is resolved by import, not by a registry."""
    module_name, _, attribute = f"{synthetic.__module__}.synthetic".rpartition(".")
    factory = getattr(importlib.import_module(module_name), attribute)

    dataset = factory(max_train_samples=2, data_dir=str(tmp_path))

    assert isinstance(dataset, SyntheticDataset)


def test_unknown_param_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="nonsense"):
        synthetic(nonsense=1, data_dir=str(tmp_path))  # type: ignore[call-arg]


def test_config_dataset_dir_name_sets_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point the default cache root at tmp_path so the test never writes to the
    # real ~/.cache/atria.
    monkeypatch.setattr(
        "atria_core.datasets._dataset._DEFAULT_ATRIA_DATASETS_CACHE_DIR", tmp_path
    )

    dataset = SyntheticDataset(config=SyntheticConfig(), dataset_dir_name="named-dir")

    assert dataset.data_dir == tmp_path / "named-dir"


def test_dataset_produces_live_split_iterators(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))

    train = dataset.split_iterator(DatasetSplitType.train)
    test = dataset.split_iterator(DatasetSplitType.test)

    assert len(train) == 4
    assert len(test) == 2
    assert _record(train[0]).sample_id == "0"


def test_dataset_repr_summarizes_data_model_and_split_sizes(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))

    representation = repr(dataset)

    assert representation.startswith("SyntheticDataset(\n")
    assert "data_model=<class " in representation
    assert "ImageInstance'>" in representation
    assert f"data_dir={tmp_path!r}" in representation
    assert "split_iterators={" in representation
    assert "IndexableSplitIterator(" in representation
    assert "<DatasetSplitType.train: 'train'>" in representation
    assert "length=4" in representation
    assert "base_iterator=" in representation
    assert "_RawSplit object" in representation
    assert "transform=" in representation
    assert "_InputTransform object" in representation
    assert "<DatasetSplitType.test: 'test'>" in representation
    assert "length=2" in representation
    assert (
        "config={'max_train_samples': None, 'max_test_samples': None}" in representation
    )


def test_dataset_repr_omits_empty_config(tmp_path: Path) -> None:
    dataset = EmptyConfigDataset(data_dir=str(tmp_path))

    assert "config=" not in repr(dataset)


def test_config_sample_caps_limit_splits(tmp_path: Path) -> None:
    dataset = SyntheticDataset(
        config=SyntheticConfig(max_train_samples=2, max_test_samples=1),
        data_dir=str(tmp_path),
    )

    assert len(dataset.train) == 2
    assert len(dataset.test) == 1
    assert [_record(sample).sample_id for sample in dataset.train] == ["0", "1"]

    snapshot = DatasetSnapshot.load(tmp_path)
    assert snapshot.splits == {"train": 2, "test": 1}


def test_cache_only_writes_configured_max_samples(tmp_path: Path) -> None:
    dataset = SyntheticDataset(
        config=SyntheticConfig(max_train_samples=2, max_test_samples=1),
        data_dir=str(tmp_path),
    )

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).cache(
        dataset, data_dir=str(tmp_path)
    )

    assert len(cached.train) == 2
    assert len(cached.test) == 1
    assert cached.snapshot.splits == {"train": 2, "test": 1}


def test_cache_then_iterate_msgpack(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).cache(
        dataset, data_dir=str(tmp_path)
    )

    train = cached.split_iterator(DatasetSplitType.train)
    test = cached.split_iterator(DatasetSplitType.test)

    assert len(train) == 4
    assert len(test) == 2
    assert {_record(sample).sample_id for sample in train} == {"0", "1", "2", "3"}


def test_cache_is_reused_on_second_call(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))
    cacher = Cacher(FileStorageType.MSGPACK, num_processes=1)
    first = cacher.cache(dataset, data_dir=str(tmp_path))

    dataset2 = SyntheticDataset(data_dir=str(tmp_path))
    second = cacher.cache(dataset2, data_dir=str(tmp_path))

    assert first.data_dir == second.data_dir


def test_process_and_cache_applies_transform_at_write_time(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))

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
    dataset = SyntheticDataset(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).process_and_cache(
        dataset, _ConfiguredTransform(prefix="configured"), data_dir=str(tmp_path)
    )

    assert cached.snapshot.transforms == [
        {
            "type": "atria_core.datasets._cacher.PreprocessTransform",
            "params": {
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


def test_cache_stores_in_memory_images_as_lazy_files(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))

    cached = Cacher(
        FileStorageType.MSGPACK,
        num_processes=1,
        store_images_to_files=True,
    ).cache(dataset, data_dir=str(tmp_path))

    image = _record(cached.train[0]).image
    assert image.content is None
    assert image.file_path is not None
    image_path = Path(image.file_path)
    assert image_path.is_file()
    assert image_path.is_relative_to(cached.data_dir / "artifacts/images")
    assert image.load().require_content().size == (4, 4)


def test_cache_with_multiprocessing_num_processes_gt_1(tmp_path: Path) -> None:
    dataset = SyntheticDataset(data_dir=str(tmp_path))

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
    dataset = SyntheticDataset(data_dir=str(tmp_path))
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
