from __future__ import annotations

import shutil

import pytest
from PIL import Image as PILImage

from atria_core.extractors import TesseractExtractorConfig
from atria_core.types import SinglePageDocumentInstance


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)
def test_tesseract_extractor_end_to_end() -> None:
    image = PILImage.new("RGB", (100, 40), color="white")
    doc = SinglePageDocumentInstance.from_image(image, sample_id="s1")
    extracted = TesseractExtractorConfig().build()(doc)

    assert isinstance(extracted, SinglePageDocumentInstance)
    assert extracted.content is not None
