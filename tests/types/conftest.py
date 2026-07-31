from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image as PILImage


@pytest.fixture
def sample_image() -> PILImage.Image:
    return PILImage.new("RGB", (16, 12), color="white")


@pytest.fixture
def sample_image_path(tmp_path: Path, sample_image: PILImage.Image) -> Path:
    path = tmp_path / "sample.png"
    sample_image.save(path)
    return path


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 100)
    pdf.new_page(200, 100)
    path = tmp_path / "sample.pdf"
    pdf.save(str(path))
    return path


def _text_page_image(text: str, size: tuple[int, int] = (400, 100)) -> PILImage.Image:
    from PIL import ImageDraw

    image = PILImage.new("RGB", size, color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, size[1] // 2 - 10), text, fill="black")
    return image


@pytest.fixture
def text_pdf_path(tmp_path: Path) -> Path:
    """A 2-page PDF where each page is a rendered image containing real,
    OCR-readable text -- for exercising the extractor end to end."""
    import pypdfium2 as pdfium

    pages_text = ["HELLO WORLD", "SECOND PAGE"]
    pdf = pdfium.PdfDocument.new()
    for text in pages_text:
        image = _text_page_image(text)
        page = pdf.new_page(*image.size)
        image_obj = pdfium.PdfImage.new(pdf)
        image_obj.set_bitmap(pdfium.PdfBitmap.from_pil(image))
        image_obj.set_matrix(pdfium.PdfMatrix().scale(*image.size))
        page.insert_obj(image_obj)
        page.gen_content()

    path = tmp_path / "text_sample.pdf"
    pdf.save(str(path))
    return path
