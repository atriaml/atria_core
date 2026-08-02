from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import Cacher, FileStorageType, IndexableSplitIterator
from atria_core.datasets._common import DatasetConfig
from atria_core.datasets._dataset import DatasetInputTransform, ImageDataset
from atria_core.registry import Registry
from atria_core.types import DatasetMetadata, DatasetSplitType, Image, ImageInstance

_synthetic = Registry.group("test_dataset_end_to_end.synthetic")


@_synthetic.register("synthetic")
@pydantic_dataclass(frozen=True)
class SyntheticConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> SyntheticDataset:
        return SyntheticDataset(self, **kwargs)


class _InputTransform(DatasetInputTransform[ImageInstance, SyntheticConfig]):
    def __call__(self, *args: Any, **kwargs: Any) -> ImageInstance:
        index: int = args[0]
        return ImageInstance(
            sample_id=str(index),
            image=Image.from_source(PILImage.new("RGB", (4, 4), color=(index, 0, 0))),
        )


class SyntheticSplitIterator(IndexableSplitIterator[ImageInstance]):
    def __init__(self, split: DatasetSplitType, data_dir: str, **kwargs: Any) -> None:
        super().__init__(split=split, **kwargs)
        self._count = 4 if split == DatasetSplitType.train else 2

    def _raw_getitem(self, index: int) -> int:
        return index

    def _raw_len(self) -> int:
        return self._count


class SyntheticDataset(ImageDataset[SyntheticConfig]):
    __input_transform__ = _InputTransform
    __split_iterator_cls__ = SyntheticSplitIterator

    def _download_urls(self) -> list[str]:
        return []

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(description="synthetic test dataset")

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]


def _record(item: object) -> ImageInstance:
    assert isinstance(item, ImageInstance)
    return item


def _mark_processed(sample: ImageInstance) -> ImageInstance:
    """Module-level (picklable) transform -- multiprocessing.Pool pickles
    task arguments, so closures/lambdas defined inside a test function
    won't survive being sent to a worker process."""
    return replace(sample, sample_id=f"processed-{sample.sample_id}")


def test_build_module_constructs_dataset(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))
    assert isinstance(dataset, SyntheticDataset)


def test_build_module_produces_live_split_iterators(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    assert len(dataset.train) == 4
    assert len(dataset.test) == 2
    assert _record(dataset.train[0]).sample_id == "0"


def test_cache_then_iterate_msgpack(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=1).cache(
        dataset, data_dir=str(tmp_path)
    )

    assert len(cached.train) == 4
    assert len(cached.test) == 2
    assert {_record(sample).sample_id for sample in cached.train} == {
        "0",
        "1",
        "2",
        "3",
    }


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

    assert {_record(sample).sample_id for sample in cached.train} == {
        "processed-0",
        "processed-1",
        "processed-2",
        "processed-3",
    }


def test_cache_with_multiprocessing_num_processes_gt_1(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module(data_dir=str(tmp_path))

    cached = Cacher(FileStorageType.MSGPACK, num_processes=2).cache(
        dataset, data_dir=str(tmp_path)
    )

    assert len(cached.train) == 4
    assert len(cached.test) == 2
    assert {_record(sample).sample_id for sample in cached.train} == {
        "0",
        "1",
        "2",
        "3",
    }


def test_cacher_validate_cache_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    assert Cacher.validate_cache(tmp_path) is False

    (tmp_path / "snapshot.yaml").write_text("dataset_class_name: Foo\n")
    assert Cacher.validate_cache(tmp_path) is False
