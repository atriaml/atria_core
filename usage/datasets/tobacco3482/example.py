"""Example: Tobacco3482 document classification dataset, end to end --
build config -> build_module() -> iterate live, then Cacher(...).cache(dataset)
-> iterate cached. `load_ocr=True` attaches real OCR content parsed from the
dataset's own pre-computed hocr files.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from random import shuffle
from typing import Any

import bs4
import numpy as np
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets import Cacher, Dataset, DatasetConfig, FileStorageType
from atria_core.logger import get_logger
from atria_core.registry import Registry
from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    ElementArray,
    OCRLevel,
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
_OCR_DATA_NAME = "tobacco3482_ocr"
_DATA_URLS = [
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_OCR_DATA_NAME}.tar.gz",
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


def _parse_hocr(hocr_path: Path) -> DocumentContent:
    """Flat, word-level parse of a tesseract hocr file -- no
    block/paragraph/line hierarchy, matching what the dataset's
    pre-computed hocr files are actually used for here."""
    with open(hocr_path) as f:
        soup = bs4.BeautifulSoup(f, features="xml")

    pages = soup.find_all("div", {"class": "ocr_page"})
    image_size_str = pages[0]["title"].split("; bbox")[1]
    width, height = (
        int(v) for v in image_size_str[4 : image_size_str.find(";")].split()
    )

    ids: list[int] = []
    bboxes: list[tuple[float, float, float, float]] = []
    texts: list[str] = []
    confs: list[float] = []
    angles: list[float] = []

    for word in soup.find_all("span", {"class": "ocrx_word"}):
        text = word.text.strip()
        if not text:
            continue

        title = word["title"]
        conf = float(title[title.find(";") + 10 :])

        angle = 0.0
        parent_title = word.parent["title"]
        if "textangle" in parent_title:
            angle = float(parent_title.split("textangle")[1][1:3])

        x1, y1, x2, y2 = (int(v) for v in title[5 : title.find(";")].split())

        ids.append(len(ids))
        bboxes.append((x1 / width, y1 / height, x2 / width, y2 / height))
        texts.append(text)
        confs.append(conf)
        angles.append(angle)

    if not ids:
        return DocumentContent(elements=None)

    elements = ElementArray(
        ids=np.array(ids),
        parent_ids=np.full(len(ids), -1),
        levels=np.full(len(ids), OCRLevel.word.value),
        bboxes=np.asarray(bboxes, dtype=np.float64),
        texts=np.asarray(texts, dtype=object),
        confs=np.asarray(confs, dtype=np.float64),
        angles=np.asarray(angles, dtype=np.float64),
    )
    return DocumentContent(elements=elements)


@datasets.register("tobacco3482")
@pydantic_dataclass(frozen=True)
class Tobacco3482Config(DatasetConfig):
    load_ocr: bool = False

    def build_module(self, **kwargs: Any) -> Tobacco3482:
        return Tobacco3482(self, **kwargs)


class InputTransform:
    def __init__(self, load_ocr: bool) -> None:
        self.load_ocr = load_ocr

    def __call__(self, input: tuple[Path, Path, int]) -> SinglePageDocumentInstance:
        image_file_path, ocr_file_path, label_index = input
        content = _parse_hocr(ocr_file_path) if self.load_ocr else None
        return SinglePageDocumentInstance.from_image(
            image_file_path, content=content
        ).add_annotation(
            ClassificationAnnotation(label_value=label_index, label_map=_CLASSES)
        )


class SplitIterator(Sequence[tuple[Path, Path, int]]):
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
        self.ocr_data_dir = Path(data_dir) / _OCR_DATA_NAME

    def __getitem__(self, index: int) -> tuple[Path, Path, int]:
        image_file_path = self.split_file_paths[index]
        label_index = _CLASSES.index(Path(image_file_path).parent.name)
        ocr_file_path = self.ocr_data_dir / image_file_path.replace(".jpg", ".hocr")
        return self.image_data_dir / image_file_path, ocr_file_path, label_index

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
        return InputTransform(self.config.load_ocr)


def main() -> None:
    dataset = Tobacco3482Config(load_ocr=True).build_module()

    train_iterator = dataset.split_iterator(DatasetSplitType.train)
    logger.info("train samples (live): %d", len(train_iterator))
    logger.info("first train sample (live): %s", train_iterator[0])

    cached = Cacher(FileStorageType.DELTALAKE).cache(dataset)

    cached_train = cached.split_iterator(DatasetSplitType.train)
    cached_test = cached.split_iterator(DatasetSplitType.test)

    logger.info("train samples (cached): %d", len(cached_train))
    logger.info("test samples (cached): %d", len(cached_test))
    logger.info("first train sample (cached): %s", cached_train[0].load())


if __name__ == "__main__":
    main()
