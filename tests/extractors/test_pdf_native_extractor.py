from __future__ import annotations

from pathlib import Path

from atria_core.extractors import PdfNativeExtractor
from atria_core.types import MultiPageDocument, OCRLevel, SinglePageDocument


def test_extracts_real_text_layer(native_text_pdf_path: Path) -> None:
    document = MultiPageDocument.from_pdf(native_text_pdf_path)
    pages = PdfNativeExtractor()(document)

    assert len(pages) == 1
    page = pages[0]
    assert isinstance(page, SinglePageDocument)
    assert page.content is not None
    assert "HELLO" in page.content.text.upper()
    assert "SECOND" in page.content.text.upper()


def test_words_grouped_under_line_segment_bboxes(native_text_pdf_path: Path) -> None:
    document = MultiPageDocument.from_pdf(native_text_pdf_path)
    page = PdfNativeExtractor()(document)[0]

    elements = page.content.elements
    words = elements.at(OCRLevel.word)
    lines = elements.at(OCRLevel.line)

    assert len(words) >= 4  # "HELLO", "WORLD", "SECOND", "LINE"
    assert len(lines) == 2

    segments = elements.segment_bboxes(OCRLevel.word)
    assert segments.shape == (len(words), 4)
    # every word's segment box should actually contain that word's own box
    word_boxes = words.bboxes
    assert (segments[:, 0] <= word_boxes[:, 0] + 1e-6).all()
    assert (segments[:, 1] <= word_boxes[:, 1] + 1e-6).all()
    assert (segments[:, 2] >= word_boxes[:, 2] - 1e-6).all()
    assert (segments[:, 3] >= word_boxes[:, 3] - 1e-6).all()


def test_populates_rasterized_image(native_text_pdf_path: Path) -> None:
    document = MultiPageDocument.from_pdf(native_text_pdf_path)
    page = PdfNativeExtractor()(document)[0]

    assert page.image is not None
    assert page.image.size[0] > 0
    assert page.image.size[1] > 0
