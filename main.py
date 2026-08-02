from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from random import Random
from typing import Any

from numpy.random import shuffle
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import (
    DatasetConfig,
)
from atria_core.datasets._dataset import Dataset
from atria_core.logger import get_logger
from atria_core.registry import Registry
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)


class Subset(Sequence):
    def __init__(
        self,
        dataset: Sequence,
        indices: Sequence[int],
    ) -> None:
        self._dataset = dataset
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        return self._dataset[self._indices[index]]


class RandomSubset(Subset):
    def __init__(
        self,
        dataset: Sequence,
        size: int,
        *,
        seed: int | None = None,
    ) -> None:
        rng = Random(seed)
        indices = rng.sample(range(len(dataset)), k=size)

        super().__init__(dataset, indices)


class MyIndexableDataset(Sequence):
    def __init__(self, data: list[int]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> int:
        return self._data[index]


class MyIterableDataset(Iterable):
    def __init__(self, data: list[int]) -> None:
        self._data = data

    def __iter__(self):
        yield from self._data


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


class SplitIterator(Sequence):
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
        self, data_dir: str, split: DatasetSplitType
    ) -> SplitIterator:
        return SplitIterator(data_dir=data_dir, split=split)

    def _build_input_transform(self) -> Callable:
        return InputTransform()


class TestTransform:
    def __call__(self, input: SinglePageDocumentInstance) -> int:
        return 1


tobacco = Tobacco3482Config().build_module()
split_iterators = tobacco._split_iterators
d = tobacco.split_iterator(DatasetSplitType.train)
for x in d.with_transform(TestTransform()):
    print(x)
    break
