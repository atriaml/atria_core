from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import FileStorageType
from atria_core.datasets._common import DatasetConfig
from atria_core.datasets._dataset import DatasetInputTransform, ImageDataset
from atria_core.registry import Registry
from atria_core.types import DatasetMetadata, DatasetSplitType, Image, ImageInstance

_synthetic = Registry.group("test_dataset_end_to_end.synthetic")


@_synthetic.register("synthetic")
@pydantic_dataclass(frozen=True)
class SyntheticConfig(DatasetConfig):
    dataset_name: str = "synthetic"

    def build_module(self) -> SyntheticDataset:
        return SyntheticDataset(self)


class _InputTransform(DatasetInputTransform[ImageInstance, SyntheticConfig]):
    def __call__(self, *args: Any, **kwargs: Any) -> ImageInstance:
        index: int = args[0]
        return ImageInstance(
            sample_id=str(index),
            image=Image.from_source(PILImage.new("RGB", (4, 4), color=(index, 0, 0))),
        )


class SyntheticDataset(ImageDataset[SyntheticConfig]):
    __input_transform__ = _InputTransform

    def _download_urls(self) -> list[str]:
        return []

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(description="synthetic test dataset")

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(self, split: DatasetSplitType, data_dir: str) -> Iterable[int]:
        count = 4 if split == DatasetSplitType.train else 2
        return list(range(count))


def _record(item: object) -> ImageInstance:
    assert isinstance(item, ImageInstance)
    return item


def test_build_module_constructs_dataset() -> None:
    dataset = SyntheticConfig().build_module()
    assert isinstance(dataset, SyntheticDataset)


def test_load_uncached_produces_live_split_iterators(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module()

    dataset.load(data_dir=str(tmp_path))

    assert len(dataset.train) == 4
    assert len(dataset.test) == 2
    assert _record(dataset.train[0]).sample_id == "0"


def test_cache_then_iterate_msgpack(tmp_path: Path) -> None:
    dataset = SyntheticConfig().build_module()

    cached = dataset.cache(
        data_dir=str(tmp_path),
        cached_storage_type=FileStorageType.MSGPACK,
        num_processes=1,
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
    dataset = SyntheticConfig().build_module()
    first = dataset.cache(
        data_dir=str(tmp_path),
        cached_storage_type=FileStorageType.MSGPACK,
        num_processes=1,
    )

    dataset2 = SyntheticConfig().build_module()
    second = dataset2.cache(
        data_dir=str(tmp_path),
        cached_storage_type=FileStorageType.MSGPACK,
        num_processes=1,
    )

    assert first.data_dir == second.data_dir
