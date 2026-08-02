"""Example: Tobacco3482 document classification dataset, end to end --
build config -> build_module() -> cache() -> iterate train/test.

Phase-1 port note: OCR loading (the old `load_ocr` config flag) isn't
carried over yet -- this only wires up the image classification path,
since porting HOCR parsing into the new DocumentContent/ElementArray
shape is separate follow-up work, not required to validate this port.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from random import shuffle
from typing import Any

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import (
    Dataset,
    DatasetConfig,
    DatasetInputTransform,
    DocumentDataset,
)
from atria_core.logger import get_logger
from atria_core.registry import Registry
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
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


class _RawSplitIterator:
    def __init__(self, split: DatasetSplitType, data_dir: str) -> None:
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

    def __iter__(self) -> Iterator[tuple[Path, int]]:
        for image_file_path in self.split_file_paths:
            label_index = _CLASSES.index(Path(image_file_path).parent.name)
            yield self.image_data_dir / image_file_path, label_index

    def __len__(self) -> int:
        return len(self.split_file_paths)


@datasets.register("tobacco3482")
@pydantic_dataclass(frozen=True)
class Tobacco3482Config(DatasetConfig):
    dataset_name: str = "tobacco3482"

    def build_module(self) -> Tobacco3482:
        return Tobacco3482(self)


class InputTransform(DatasetInputTransform[DocumentInstance, Tobacco3482Config]):
    def __call__(self, *args: Any, **kwargs: Any) -> DocumentInstance:
        image_file_path, label_index = args[0]
        return SinglePageDocumentInstance.from_image(image_file_path).add_annotation(
            ClassificationAnnotation(label=label_index, label_map=_CLASSES)
        )


class Tobacco3482(DocumentDataset[Tobacco3482Config]):
    __input_transform__ = InputTransform

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

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Iterable[tuple[Path, int]]:
        return _RawSplitIterator(split=split, data_dir=data_dir)


def main() -> None:
    dataset: Dataset[Tobacco3482Config, DocumentInstance] = (
        Tobacco3482Config().build_module()
    )
    cached = dataset.cache()  # defaults to Delta Lake storage

    logger.info("train samples: %d", len(cached.train))
    logger.info("test samples: %d", len(cached.test))

    sample = cached.train[0]
    logger.info("first train sample: %s", sample)


if __name__ == "__main__":
    main()
