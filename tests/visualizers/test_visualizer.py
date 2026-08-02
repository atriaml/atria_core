from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image as PILImage

from atria_core.types import (
    BaseDataInstance,
    DocumentContent,
    ElementArray,
    ImageInstance,
    MultiPageDocumentInstance,
    SinglePageDocumentInstance,
)
from atria_core.visualizers import (
    visualize,
    visualize_document_instance,
    visualize_image_instance,
)
from tests.types.builders import (
    make_classification_annotation,
    make_document_content,
    make_image,
    make_object_detection_annotation,
)


def test_visualize_image_instance_saves_png(tmp_path: Path) -> None:
    instance = ImageInstance(sample_id="s1", image=make_image())

    path = visualize_image_instance(instance, str(tmp_path))

    assert path == tmp_path / "s1.png"
    assert path.exists()
    assert PILImage.open(path).size == make_image().require_content().size


def test_visualize_image_instance_draws_object_detection_boxes(tmp_path: Path) -> None:
    image = make_image(source=PILImage.new("RGB", (100, 100), color="white"))
    detection = make_object_detection_annotation(
        bboxes=np.array([[0.1, 0.1, 0.5, 0.5]]), normalized=True
    )
    instance = ImageInstance(sample_id="s1", image=image).add_annotation(detection)

    path = visualize_image_instance(instance, str(tmp_path))

    saved = np.array(PILImage.open(path))
    # a box was drawn somewhere -- image is no longer solid white.
    assert not np.all(saved == 255)


def test_visualize_image_instance_output_name_includes_classification_label(
    tmp_path: Path,
) -> None:
    instance = ImageInstance(sample_id="s1", image=make_image()).add_annotation(
        make_classification_annotation(label=1)
    )

    path = visualize_image_instance(instance, str(tmp_path))

    assert path.name == "s1_label=dog.png"


def test_visualize_document_instance_image_sourced_saves_png(tmp_path: Path) -> None:
    instance = SinglePageDocumentInstance.from_image(
        make_image().require_content(),
        sample_id="d1",
        content=make_document_content(),
    )

    path = visualize_document_instance(instance, str(tmp_path))

    assert path == tmp_path / "d1.png"
    assert path.exists()


def test_visualize_document_instance_pdf_page_draws_on_pdf_directly(
    tmp_path: Path, sample_pdf_path: Path
) -> None:
    elements = ElementArray.from_words(
        ["hello", "world"], [[0.1, 0.1, 0.3, 0.2], [0.35, 0.1, 0.6, 0.2]]
    )
    instance = SinglePageDocumentInstance.from_pdf(
        sample_pdf_path, page_id=0, content=DocumentContent(elements=elements)
    )

    path = visualize_document_instance(instance, str(tmp_path))

    assert path == tmp_path / f"{instance.sample_id}.pdf"
    reopened = pymupdf.open(str(path))
    assert len(reopened) == 1
    assert len(reopened[0].get_drawings()) > 0


def test_visualize_document_instance_multi_page_reexports_all_pages(
    tmp_path: Path, sample_pdf_path: Path
) -> None:
    instance = MultiPageDocumentInstance(
        sample_id="m1", source_path=str(sample_pdf_path)
    )

    path = visualize_document_instance(instance, str(tmp_path))

    assert path == tmp_path / "m1.pdf"
    reopened = pymupdf.open(str(path))
    assert len(reopened) == instance.num_pages


def test_visualize_dispatches_by_instance_type(tmp_path: Path) -> None:
    image_instance = ImageInstance(sample_id="s1", image=make_image())
    assert visualize(image_instance, str(tmp_path)) == tmp_path / "s1.png"

    document_instance = SinglePageDocumentInstance.from_image(
        make_image().require_content(), sample_id="d1"
    )
    assert visualize(document_instance, str(tmp_path)) == tmp_path / "d1.png"


def test_visualize_raises_for_unsupported_instance_type(tmp_path: Path) -> None:
    instance = BaseDataInstance(sample_id="s1")
    with pytest.raises(TypeError, match="No visualizer"):
        visualize(instance, str(tmp_path))
