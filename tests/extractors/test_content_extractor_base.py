from __future__ import annotations

from PIL import Image as PILImage

from atria_core.extractors import ContentExtractor
from atria_core.types import DocumentContent, SinglePageDocument


class StubExtractor(ContentExtractor):
    def _extract(self, image: PILImage.Image) -> DocumentContent:
        return DocumentContent(_text=f"{image.width}x{image.height}")


def test_extract_content_via_extractor_call(sample_image: PILImage.Image) -> None:
    doc = SinglePageDocument.from_image(sample_image)
    extracted = StubExtractor()(doc)
    assert isinstance(extracted, SinglePageDocument)
    assert extracted.content is not None
    assert extracted.content.text == "16x12"
    # original document is untouched
    assert doc.content is None
