from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image as PILImage

from atria_core.types._data_instance._document_instance import (
    DocumentInstance,
    MultiPageDocumentInstance,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._documents import PdfPage
from atria_core.types._generic._image import Image


def test_single_page_from_image_derives_sample_id_from_path(
    sample_image_path: Path,
) -> None:
    instance = SinglePageDocumentInstance.from_image(sample_image_path)
    assert instance.sample_id == sample_image_path.name
    assert isinstance(instance.visual, Image)
    assert instance.page_id is None


def test_single_page_from_image_requires_explicit_sample_id_for_pil_image(
    sample_image: PILImage.Image,
) -> None:
    with pytest.raises(AssertionError):
        SinglePageDocumentInstance.from_image(sample_image)


def test_single_page_from_image_with_explicit_sample_id(
    sample_image: PILImage.Image,
) -> None:
    instance = SinglePageDocumentInstance.from_image(sample_image, sample_id="s1")
    assert instance.sample_id == "s1"


def test_single_page_from_pdf_derives_sample_id_with_page_id(
    sample_pdf_path: Path,
) -> None:
    instance = SinglePageDocumentInstance.from_pdf(sample_pdf_path, page_id=1)
    assert instance.sample_id == f"{sample_pdf_path.name}#1"
    assert isinstance(instance.visual, PdfPage)
    assert instance.page_id == 1


def test_single_page_load_delegates_to_visual(sample_image_path: Path) -> None:
    instance = SinglePageDocumentInstance.from_image(sample_image_path)
    loaded = instance.load()
    assert loaded.require_content() is not None


def test_single_page_to_dict_from_dict_roundtrip_image(
    sample_image_path: Path,
) -> None:
    instance = SinglePageDocumentInstance.from_image(sample_image_path)
    data = instance.to_dict()
    assert data["visual_type"] == "image"
    restored = SinglePageDocumentInstance.from_dict(data)
    assert restored.sample_id == instance.sample_id
    assert isinstance(restored.visual, Image)


def test_single_page_to_dict_from_dict_roundtrip_pdf_page(
    sample_pdf_path: Path,
) -> None:
    instance = SinglePageDocumentInstance.from_pdf(sample_pdf_path, page_id=0)
    data = instance.to_dict()
    assert data["visual_type"] == "pdf_page"
    restored = SinglePageDocumentInstance.from_dict(data)
    assert restored.sample_id == instance.sample_id
    assert isinstance(restored.visual, PdfPage)
    assert restored.visual.page_id == 0


def test_single_page_equality() -> None:
    a = SinglePageDocumentInstance.from_image("/some/path.png", sample_id="s1")
    b = SinglePageDocumentInstance.from_image("/some/path.png", sample_id="s1")
    c = SinglePageDocumentInstance.from_image("/some/path.png", sample_id="s2")
    assert a == b
    assert a != c


def test_multi_page_from_pdf_num_pages(sample_pdf_path: Path) -> None:
    instance = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    assert instance.num_pages == 2


def test_multi_page_get_page_returns_single_page_instance_with_suffixed_id(
    sample_pdf_path: Path,
) -> None:
    instance = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    page = instance.get_page(0)
    assert isinstance(page, SinglePageDocumentInstance)
    assert page.sample_id == f"{sample_pdf_path.name}#0"
    assert isinstance(page.visual, PdfPage)
    assert page.visual.content is None


def test_multi_page_get_page_out_of_range_raises(sample_pdf_path: Path) -> None:
    instance = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    with pytest.raises(AssertionError):
        instance.get_page(5)


def test_multi_page_iteration_yields_all_pages(sample_pdf_path: Path) -> None:
    instance = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    pages = list(instance)
    assert [p.sample_id for p in pages] == [
        f"{sample_pdf_path.name}#0",
        f"{sample_pdf_path.name}#1",
    ]


def test_multi_page_page_carries_its_own_annotation(sample_pdf_path: Path) -> None:
    from atria_core.types._generic._annotations import (
        AnnotationType,
        ClassificationAnnotation,
    )

    instance = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    page0 = instance.get_page(0).add_annotation(
        ClassificationAnnotation(label_value=0, label_name="a")
    )
    instance = MultiPageDocumentInstance(
        sample_id=instance.sample_id,
        pages=[page0, instance.get_page(1)],
    )

    assert instance.get_page(0).has_annotation_type(AnnotationType.classification)
    assert not instance.get_page(1).has_annotation_type(AnnotationType.classification)


def test_multi_page_deck_wide_annotation(sample_pdf_path: Path) -> None:
    from atria_core.types._generic._annotations import (
        AnnotationType,
        ClassificationAnnotation,
    )

    instance = MultiPageDocumentInstance.from_pdf(
        sample_pdf_path, sample_id="m1"
    ).add_annotation(ClassificationAnnotation(label_value=0, label_name="a"))

    assert instance.has_annotation_type(AnnotationType.classification)


def test_multi_page_load_loads_every_page(sample_pdf_path: Path) -> None:
    instance = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    assert instance.get_page(0).visual.content is None

    loaded = instance.load()

    assert all(page.visual.content is not None for page in loaded)


def test_multi_page_to_dict_from_dict_roundtrip(sample_pdf_path: Path) -> None:
    instance = MultiPageDocumentInstance.from_pdf(
        sample_pdf_path, sample_id="m1", dpi=150
    )
    data = instance.to_dict()
    restored = MultiPageDocumentInstance.from_dict(data)
    assert restored.sample_id == "m1"
    assert restored.num_pages == instance.num_pages
    assert restored.get_page(0).visual.dpi == 150


def test_multi_page_equality(sample_pdf_path: Path) -> None:
    a = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    b = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    c = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1", dpi=100)
    assert a == b
    assert a != c


def test_single_and_multi_page_are_both_document_instances(
    sample_image_path: Path, sample_pdf_path: Path
) -> None:
    single = SinglePageDocumentInstance.from_image(sample_image_path)
    multi = MultiPageDocumentInstance.from_pdf(sample_pdf_path, sample_id="m1")
    assert isinstance(single, DocumentInstance)
    assert isinstance(multi, DocumentInstance)
