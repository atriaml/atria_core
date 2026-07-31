import pdfplumber
from PIL.Image import Image as PILImage

from atria_core.logger import get_logger
from atria_core.types._base._data_model import BaseDataModel
from atria_core.types._generic._bounding_box import BoundingBox, BoundingBoxMode
from atria_core.types._generic._doc_content import TextElement
from atria_core.types._pydantic import OptIntField, OptStrField

logger = get_logger(__name__)


class PDF(BaseDataModel):
    file_path: OptStrField = None
    num_pages: OptIntField = None

    @property
    def pages(self) -> list[PILImage]:
        """Lazily extract pages from PDF when accessed."""
        if self.file_path is None:
            raise ValueError(
                "PDF file path is not set. Please set file_path before accessing pages."
            )

        from pdf2image import convert_from_path

        return convert_from_path(self.file_path)

    def get_page(self, page_num: int) -> PILImage:
        """Get a specific page from the PDF (0-indexed)."""
        if page_num < 0 or (self.num_pages is not None and page_num >= self.num_pages):
            raise IndexError(
                f"Page {page_num} is out of range. PDF has {self.num_pages} pages."
            )

        return self.pages[page_num]

    def load(self):
        """Load PDF metadata without extracting pages."""
        if self.file_path is None:
            raise ValueError(
                "PDF file path is not set. Please set file_path before loading."
            )

        # Load number of pages if not already set
        if self.num_pages is None:
            try:
                import pymupdf

                doc = pymupdf.open(self.file_path)
                num_pages = doc.page_count
                doc.close()

                return PDF(
                    file_path=self.file_path,
                    num_pages=num_pages,
                )
            except ImportError:
                # Fallback to pdf2image if pymupdf is not available or fails
                from pdf2image import convert_from_path

                pages = convert_from_path(self.file_path)
                num_pages = len(pages)

                return PDF(
                    file_path=self.file_path,
                    num_pages=num_pages,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load PDF metadata from {self.file_path}: {e}"
                ) from e
        return self

    def extract_text_elements(self, page_number: int = 0) -> list["TextElement"]:
        assert self.file_path is not None, (
            "PDF file path must be set to extract text elements."
        )
        text_elements: list[TextElement] = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
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
            logger.exception(f"Error reading PDF {self.file_path}: {e}")
        return text_elements
