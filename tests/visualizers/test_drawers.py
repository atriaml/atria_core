from __future__ import annotations

import numpy as np
import pymupdf
from PIL import Image as PILImage

from atria_core.visualizers import ImageDrawer, PdfDrawer


def test_image_drawer_draws_in_place_and_returns_same_image() -> None:
    image = PILImage.new("RGB", (100, 100), color="white")
    bboxes = np.array([[10.0, 10.0, 50.0, 50.0]])

    result = ImageDrawer().draw(image, bboxes, texts=["hello"], labels=["word"])

    assert result is image
    assert not np.all(np.array(image) == 255)


def test_pdf_drawer_draws_in_place_and_returns_same_page() -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    bboxes = np.array([[10.0, 10.0, 80.0, 30.0]])

    result = PdfDrawer().draw(page, bboxes, texts=["hello"], labels=["word"])

    assert result is page
    assert len(page.get_drawings()) > 0
