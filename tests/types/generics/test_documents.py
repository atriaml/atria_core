from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from atria_core.types import MultiPageDocument, SinglePageDocument


def test_single_page_document_from_pil_image(sample_image: PILImage.Image) -> None:
    doc = SinglePageDocument.from_image(sample_image)
    assert doc.source_path is None
    assert doc.image is sample_image
    assert doc.page_id is None
    assert doc.content is None


def test_single_page_document_from_path(sample_image_path: Path) -> None:
    doc = SinglePageDocument.from_image(sample_image_path, page_id=0)
    assert doc.source_path == str(sample_image_path)
    assert doc.page_id == 0
    assert doc.image.size == (16, 12)


def test_multi_page_document_num_pages(sample_pdf_path: Path) -> None:
    doc = MultiPageDocument.from_pdf(sample_pdf_path)
    assert doc.num_pages == 2


def test_multi_page_document_get_page(sample_pdf_path: Path) -> None:
    doc = MultiPageDocument.from_pdf(sample_pdf_path)
    page = doc.get_page(0)
    assert isinstance(page, SinglePageDocument)
    assert page.page_id == 0
    assert page.source_path == str(sample_pdf_path)


def test_multi_page_document_iteration(sample_pdf_path: Path) -> None:
    doc = MultiPageDocument.from_pdf(sample_pdf_path)
    pages = list(doc)
    assert len(pages) == 2
    assert [p.page_id for p in pages] == [0, 1]


def test_multi_page_document_equality(sample_pdf_path: Path) -> None:
    a = MultiPageDocument.from_pdf(sample_pdf_path)
    b = MultiPageDocument.from_pdf(sample_pdf_path)
    c = MultiPageDocument.from_pdf(sample_pdf_path, dpi=100)
    assert a == b
    assert a != c
