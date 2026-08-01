from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from atria_core.serialization import DatasetReader, DatasetWriter
from atria_core.types import (
    AnnotationType,
    DocumentInstance,
    Image,
    ImageInstance,
    MultiPageDocument,
    SinglePageDocument,
)
from tests.types.builders import make_classification_annotation


def test_image_instance_dataset_roundtrip(tmp_path: Path) -> None:
    instances = [
        ImageInstance(
            sample_id="s1",
            image=Image.from_source(PILImage.new("RGB", (8, 6), color="red")),
        ).add_annotation(make_classification_annotation()),
        ImageInstance(
            sample_id="s2", image=Image.from_source(PILImage.new("RGB", (8, 6), color="blue"))
        ),
    ]

    DatasetWriter(tmp_path).write(instances)
    assert (tmp_path / "data.parquet").exists()
    assert (tmp_path / "artifacts" / "s1.png").exists()
    assert (tmp_path / "artifacts" / "s2.png").exists()

    restored = list(DatasetReader(tmp_path))
    assert len(restored) == 2

    for original, loaded in zip(instances, restored, strict=True):
        assert loaded.sample_id == original.sample_id
        loaded_image = loaded.image.load()
        assert np.array_equal(
            np.array(original.image.require_content()),
            np.array(loaded_image.require_content()),
        )

    restored_ann = restored[0].get_annotation_by_type(AnnotationType.classification)
    assert restored_ann is not None
    assert restored_ann.label_name == "cat"
    assert restored[1].get_annotation_by_type(AnnotationType.classification) is None


def test_document_instance_single_page_dataset_roundtrip(tmp_path: Path) -> None:
    original_image = PILImage.new("RGB", (10, 5), color="green")
    document = SinglePageDocument.from_image(original_image)
    instances = [DocumentInstance(sample_id="d1", document=document)]

    DatasetWriter(tmp_path).write(instances)
    assert (tmp_path / "artifacts" / "d1.png").exists()

    restored = list(DatasetReader(tmp_path))
    assert len(restored) == 1
    loaded = restored[0]
    assert isinstance(loaded.document, SinglePageDocument)
    assert np.array_equal(np.array(original_image), np.array(loaded.document.image))


def test_document_instance_multi_page_dataset_roundtrip(
    tmp_path: Path, sample_pdf_path: Path
) -> None:
    document = MultiPageDocument.from_pdf(sample_pdf_path)
    instances = [DocumentInstance(sample_id="m1", document=document)]

    out_dir = tmp_path / "dataset"
    DatasetWriter(out_dir).write(instances)
    assert (out_dir / "artifacts" / "m1.pdf").exists()

    restored = list(DatasetReader(out_dir))
    assert len(restored) == 1
    loaded = restored[0]
    assert isinstance(loaded.document, MultiPageDocument)
    assert loaded.document.num_pages == document.num_pages


def test_dataset_roundtrip_preserves_sample_order(tmp_path: Path) -> None:
    instances = [
        ImageInstance(sample_id=f"s{i}", image=Image.from_source(PILImage.new("RGB", (4, 4))))
        for i in range(5)
    ]
    DatasetWriter(tmp_path).write(instances)
    restored = list(DatasetReader(tmp_path))
    assert [r.sample_id for r in restored] == [i.sample_id for i in instances]
