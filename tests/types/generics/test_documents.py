from __future__ import annotations

from pathlib import Path

import pytest

from atria_core.types._generic._documents import PdfPage


def test_construct_from_path_is_lazy(sample_pdf_path: Path) -> None:
    page = PdfPage(file_path=str(sample_pdf_path), page_id=0)
    assert page.file_path == str(sample_pdf_path)
    assert page.content is None


def test_require_content_without_load_raises(sample_pdf_path: Path) -> None:
    page = PdfPage(file_path=str(sample_pdf_path), page_id=0)
    with pytest.raises(AssertionError):
        page.require_content()


def test_load_renders_page(sample_pdf_path: Path) -> None:
    page = PdfPage(file_path=str(sample_pdf_path), page_id=0)
    loaded = page.load()
    assert loaded.content is not None
    assert loaded.file_path == page.file_path
    assert loaded.page_id == 0


def test_load_is_idempotent_when_already_loaded(sample_pdf_path: Path) -> None:
    page = PdfPage(file_path=str(sample_pdf_path), page_id=0).load()
    loaded = page.load()
    assert loaded is page


def test_to_dict_from_dict_roundtrip_via_file_path(sample_pdf_path: Path) -> None:
    page = PdfPage(file_path=str(sample_pdf_path), page_id=1, dpi=150)
    data = page.to_dict()
    assert data == {"file_path": str(sample_pdf_path), "page_id": 1, "dpi": 150}
    restored = PdfPage.from_dict(data)
    assert restored.file_path == str(sample_pdf_path)
    assert restored.page_id == 1
    assert restored.dpi == 150
    assert restored.content is None


def test_to_dict_from_dict_roundtrip_via_embedded_bytes(sample_pdf_path: Path) -> None:
    page = PdfPage(file_path=str(sample_pdf_path), page_id=0).load()
    data = page.to_dict()
    assert "content_bytes" in data
    assert "file_path" not in data
    restored = PdfPage.from_dict(data)
    assert restored.file_path is None
    assert restored.content is not None
    assert restored.page_id == 0


def test_to_dict_without_path_or_content_raises() -> None:
    with pytest.raises(AssertionError):
        PdfPage().to_dict()


def test_equality_by_field() -> None:
    a = PdfPage(file_path="/some/doc.pdf", page_id=0)
    b = PdfPage(file_path="/some/doc.pdf", page_id=0)
    c = PdfPage(file_path="/some/doc.pdf", page_id=1)
    assert a == b
    assert a != c
