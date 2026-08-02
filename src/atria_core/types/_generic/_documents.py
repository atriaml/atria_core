from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from PIL import Image as PILImage

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._utilities._url_fetchers import ResourceLoader


@dataclass(frozen=True, repr=False)
class PdfPage(BaseDataModel):
    """One page of a PDF, rendered lazily -- same load()/require_content()
    shape as Image, so a document instance can hold either interchangeably."""

    file_path: str | None = None
    page_id: int = 0
    dpi: int = 200
    content: PILImage.Image | None = None

    def load(self) -> PdfPage:
        if self.content is not None:
            return self

        import pypdfium2 as pdfium

        assert self.file_path is not None, "PdfPage has neither content nor file_path"
        pdf_bytes = ResourceLoader.for_uri(self.file_path).load_bytes()
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            page = pdf[self.page_id]
            try:
                bitmap = page.render(scale=self.dpi / 72)
                image = bitmap.to_pil()
            finally:
                page.close()
        finally:
            pdf.close()
        return replace(self, content=image)

    def require_content(self) -> PILImage.Image:
        assert self.content is not None, (
            "PdfPage content is not loaded; call load() first"
        )
        return self.content

    def to_dict(self) -> dict[str, Any]:
        if self.content is not None:
            from atria_core.types._utilities._image_encoding import _image_to_bytes

            return {
                "content_bytes": _image_to_bytes(self.content),
                "page_id": self.page_id,
                "dpi": self.dpi,
            }
        assert self.file_path is not None, "PdfPage has neither content nor file_path"
        return {"file_path": self.file_path, "page_id": self.page_id, "dpi": self.dpi}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PdfPage:
        if "content_bytes" in data:
            from atria_core.types._utilities._image_encoding import _bytes_to_image

            return cls(
                page_id=data.get("page_id", 0),
                dpi=data.get("dpi", 200),
                content=_bytes_to_image(data["content_bytes"]),
            )
        return cls(
            file_path=data["file_path"],
            page_id=data.get("page_id", 0),
            dpi=data.get("dpi", 200),
        )
