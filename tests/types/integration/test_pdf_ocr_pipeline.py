from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from atria_core.types._data_instance._document_instance import DocumentInstance
from atria_core.types._extractors._tesseract import TesseractExtractorConfig
from atria_core.types._generic._documents import MultiPageDocument

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)


def test_local_pdf_to_ocr_populated_document_instances(text_pdf_path: Path) -> None:
    """End-to-end use case B: user points a dataset at a local PDF, extracts
    OCR per page, and gets back DocumentInstances with content populated --
    no atria_datasets involved, just the pieces atria_core provides."""
    document = MultiPageDocument.from_pdf(text_pdf_path)
    extractor = TesseractExtractorConfig()

    instances = [
        DocumentInstance(
            sample_id=f"{text_pdf_path.stem}_page_{page.page_id}",
            document=page.extract_content(extractor),
        )
        for page in document
    ]

    assert len(instances) == 2
    for instance in instances:
        assert instance.document.content is not None
        assert instance.document.content.text is not None

    recognized = " ".join(i.document.content.text.upper() for i in instances)
    assert "HELLO" in recognized
    assert "WORLD" in recognized
    assert "SECOND" in recognized
    assert "PAGE" in recognized


def test_single_page_pdf_source_ocr_roundtrip(text_pdf_path: Path) -> None:
    """Same flow but going through a single page pulled directly off the
    MultiPageDocument, mirroring how a dataset would process one sample."""
    document = MultiPageDocument.from_pdf(text_pdf_path)
    page = document.get_page(0)

    extracted = page.extract_content(TesseractExtractorConfig())

    assert extracted.content is not None
    assert "HELLO" in extracted.content.text.upper()
    # bboxes come back normalized to [0, 1] by the extractor
    assert extracted.content.normalized is True
    assert len(extracted.content.bbox_list) > 0
    for bbox in extracted.content.bbox_list:
        assert (bbox >= 0).all()
        assert (bbox <= 1).all()
