"""Example: Tobacco3482 document classification dataset, end to end --
build config -> build_module() -> iterate live, then Cacher(...).cache(dataset)
-> iterate cached. `load_ocr=True` attaches real OCR content parsed from the
dataset's own pre-computed hocr files.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import Cacher, FileStorageType
from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.logger import get_logger
from atria_core.registry import Registry
from atria_core.types import (
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._data_instance._image_instance import ImageInstance
from atria_core.types._generic._annotations import ClassificationAnnotation
from atria_core.types._generic._image import Image

logger = get_logger(__name__)

datasets = Registry.group("datasets")


@datasets.register("mnist")
@pydantic_dataclass(frozen=True)
class MNISTConfig(HuggingfaceDatasetConfig):
    config_name: str = "mnist"

    def build_module(self) -> MNIST:
        return MNIST("ylecun/mnist", config=self)


class InputTransform:
    def __init__(self, labels: list[str]):
        self._labels = labels

    def __call__(self, sample) -> ImageInstance:
        return ImageInstance(
            sample_id=str(uuid.uuid4()),
            image=Image(content=sample["image"]),
        ).add_annotation(
            ClassificationAnnotation(
                label_value=sample["label"],
                label_name=self._labels[sample["label"]],
            )
        )


class MNIST(HuggingfaceDataset[MNISTConfig, ImageInstance]):
    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform(labels=self.metadata.dataset_labels.classification)


def main() -> None:
    dataset = MNISTConfig().build_module()
    cached = Cacher(FileStorageType.MSGPACK).cache(dataset)

    cached_train = cached.split_iterator(DatasetSplitType.train)
    cached_test = cached.split_iterator(DatasetSplitType.test)

    logger.info("train samples (cached): %d", len(cached_train))
    logger.info("test samples (cached): %d", len(cached_test))
    logger.info("first train sample (cached): %s", cached_train[0].load())


if __name__ == "__main__":
    main()
