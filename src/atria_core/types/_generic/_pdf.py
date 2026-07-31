from __future__ import annotations

import io
from typing import Self

import pdfplumber

from atria_core.logger import get_logger
from atria_core.types._base._data_model import BaseDataModel
from atria_core.types._generic._bounding_box import BoundingBox, BoundingBoxMode
from atria_core.types._generic._doc_content import TextElement
from atria_core.types._generic._image import Image

logger = get_logger(__name__)


class PDF(BaseDataModel):
    content: bytes

    @classmethod
    def from_file_path_or_uri(cls, file_uri: str) -> Self:
        from atria_core.types._utilities._url_fetchers import ResourceLoader

        return cls(content=ResourceLoader.for_uri(file_uri).load_bytes())

    def get_page(self, page_number: int) -> Image:
        """Render a specific page (0-indexed) as an Image."""
        from pdf2image import convert_from_bytes

        pil_image = convert_from_bytes(
            self.content, first_page=page_number + 1, last_page=page_number + 1
        )[0]
        return Image(content=pil_image)

    def extract_text_elements(self, page_number: int = 0) -> list[TextElement]:
        text_elements: list[TextElement] = []
        try:
            with pdfplumber.open(io.BytesIO(self.content)) as pdf:
                page = pdf.pages[page_number]
                page_width = page.width
                page_height = page.height

                # Get crop box coordinates
                crop = page.cropbox  # returns (x0, y0, x1, y1)
                crop_x0, crop_y0, crop_x1, crop_y1 = crop
                crop_width = crop_x1 - crop_x0
                crop_height = crop_y1 - crop_y0
                crop_x0 = max(0, crop_x0)
                crop_y0 = max(0, crop_y0)
                crop_x1 = min(page_width, crop_x1)
                crop_y1 = min(page_height, crop_y1)

                for w in page.within_bbox(
                    (crop_x0, crop_y0, crop_x1, crop_y1)
                ).extract_words(x_tolerance=1, y_tolerance=1):
                    text = w["text"].strip()
                    if not text:
                        continue

                    # Adjust coordinates relative to crop box
                    x0 = w["x0"] - crop_x0
                    top = w["top"] - crop_y0
                    x1 = w["x1"] - crop_x0
                    bottom = w["bottom"] - crop_y0

                    bbox = BoundingBox(
                        value=[x0, top, x1, bottom], mode=BoundingBoxMode.XYXY
                    ).ops.normalize(width=crop_width, height=crop_height)
                    text_elements.append(TextElement(text=text, bbox=bbox))
        except Exception as e:
            logger.exception(f"Error reading PDF page {page_number}: {e}")
        return text_elements
