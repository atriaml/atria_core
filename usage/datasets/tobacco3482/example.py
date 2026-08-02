"""Example: Tobacco3482 document classification dataset, end to end --
build config -> build_module() -> iterate live, then Cacher(...).cache(dataset)
-> iterate cached.

Phase-1 port note: OCR loading (the old `load_ocr` config flag) isn't
carried over yet -- this only wires up the image classification path,
since porting HOCR parsing into the new DocumentContent/ElementArray
shape is separate follow-up work, not required to validate this port.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from random import shuffle
from typing import Any

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import Cacher, Dataset, DatasetConfig, FileStorageType
from atria_core.logger import get_logger
from atria_core.registry import Registry
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)

logger = get_logger(__name__)

datasets = Registry.group("datasets")

_CITATION = """\
@article{Kumar2014StructuralSF,
    title={Structural similarity for document image classification and retrieval},
    author={Jayant Kumar and Peng Ye and David S. Doermann},
    journal={Pattern Recognit. Lett.},
    year={2014},
    volume={43},
    pages={119-126}
}
"""
_DESCRIPTION = (
    "The Tobacco3482 dataset consists of 3842 grayscale images in 10 classes. "
    "In this version, the dataset is split into 2782 training images, and 700 "
    "test images."
)
_HOMEPAGE = "https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg"
_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"
_IMAGE_DATA_NAME = "tobacco3482"
_DATA_URLS = [
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    "https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/train.txt",
    "https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/test.txt",
]
_CLASSES = [
    "Letter",
    "Resume",
    "Scientific",
    "ADVE",
    "Email",
    "Report",
    "News",
    "Memo",
    "Form",
    "Note",
]


@datasets.register("tobacco3482")
@pydantic_dataclass(frozen=True)
class Tobacco3482Config(DatasetConfig):
    def build_module(self, **kwargs: Any) -> Tobacco3482:
        return Tobacco3482(self, **kwargs)


class InputTransform:
    def __call__(self, input: tuple[Path, int]) -> SinglePageDocumentInstance:
        image_file_path, label_index = input
        return SinglePageDocumentInstance.from_image(image_file_path).add_annotation(
            ClassificationAnnotation(label=label_index, label_map=_CLASSES)
        )


class SplitIterator(Sequence[tuple[Path, int]]):
    def __init__(self, data_dir: str, split: DatasetSplitType) -> None:
        if split == DatasetSplitType.train:
            split_file_path = Path(data_dir) / "train.txt"
        elif split == DatasetSplitType.test:
            split_file_path = Path(data_dir) / "test.txt"
        else:
            raise ValueError(f"Unsupported split: {split}")
        with open(split_file_path) as f:
            self.split_file_paths = f.read().splitlines()
            shuffle(self.split_file_paths)
        self.image_data_dir = Path(data_dir) / _IMAGE_DATA_NAME

    def __getitem__(self, index: int) -> tuple[Path, int]:
        image_file_path = self.split_file_paths[index]
        label_index = _CLASSES.index(Path(image_file_path).parent.name)
        return self.image_data_dir / image_file_path, label_index

    def __len__(self) -> int:
        return len(self.split_file_paths)


class Tobacco3482(Dataset[Tobacco3482Config, SinglePageDocumentInstance]):
    def _download_urls(self) -> list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()


def main() -> None:
    dataset = Tobacco3482Config().build_module()

    train_iterator = dataset.split_iterator(DatasetSplitType.train)
    logger.info("train samples (live): %d", len(train_iterator))
    logger.info("first train sample (live): %s", train_iterator[0])

    cached = Cacher(FileStorageType.DELTALAKE).cache(dataset)

    cached_train = cached.split_iterator(DatasetSplitType.train)
    cached_test = cached.split_iterator(DatasetSplitType.test)
    logger.info("train samples (cached): %d", len(cached_train))
    logger.info("test samples (cached): %d", len(cached_test))
    logger.info("first train sample (cached): %s", cached_train[0])


if __name__ == "__main__":
    main()
