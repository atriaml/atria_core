"""Example: MNIST from the Hugging Face hub, end to end -- construct the
dataset (default config, or an explicit one) -> iterate live, then
Cacher(...).cache(dataset) -> iterate cached.

Configs describe params; they never build anything. The dataset takes one, and
constructs its generic config type when none is given. Import the class to get
the exact type, or build it by name with `datasets.create("mnist")`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from atria_core.datasets import Cacher, FileStorageType, datasets
from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType
from atria_core.types._data_instance._image_instance import ImageInstance
from atria_core.types._generic._annotations import ClassificationAnnotation
from atria_core.types._generic._image import Image

logger = get_logger(__name__)

_REPO = "ylecun/mnist"


class MNISTConfig(HuggingfaceDatasetConfig):
    config_name: str | None = "mnist"


class InputTransform:
    def __init__(self, labels: list[str]):
        self._labels = labels

    def __call__(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            sample_id=str(uuid.uuid4()),
            image=Image(content=sample["image"]),
        ).add_annotation(
            ClassificationAnnotation(
                label_value=sample["label"],
                label_name=self._labels[sample["label"]],
            )
        )


@datasets.register("mnist")
class MNIST(HuggingfaceDataset[ImageInstance, MNISTConfig]):
    def __init__(self, *, config: MNISTConfig | None = None, **kwargs: Any) -> None:
        super().__init__(repo=_REPO, config=config, **kwargs)

    def _build_input_transform(self) -> Callable[[Any], ImageInstance]:
        return InputTransform(labels=self.metadata.dataset_labels.classification)


def main() -> None:
    dataset = MNIST()
    cached = Cacher(FileStorageType.MSGPACK).cache(dataset)

    cached_train = cached.split_iterator(DatasetSplitType.train)
    cached_test = cached.split_iterator(DatasetSplitType.test)

    logger.info("train samples (cached): %d", len(cached_train))  # type: ignore
    logger.info("test samples (cached): %d", len(cached_test))  # type: ignore
    logger.info("first train sample (cached): %s", cached_train[0].load())  # type: ignore


if __name__ == "__main__":
    main()
