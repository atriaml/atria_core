from __future__ import annotations

import shutil

import pytest
from PIL import Image as PILImage

from atria_core.extractors import TesseractExtractorConfig
from atria_core.types import SinglePageDocument


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)
def test_tesseract_extractor_end_to_end() -> None:
    image = PILImage.new("RGB", (100, 40), color="white")
    doc = SinglePageDocument.from_image(image)
    extracted = TesseractExtractorConfig().build()(doc)

    assert isinstance(extracted, SinglePageDocument)
    assert extracted.content is not None
