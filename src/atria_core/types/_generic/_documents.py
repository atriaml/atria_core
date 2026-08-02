from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._doc_content import DocumentContent
from atria_core.types._utilities._url_fetchers import (
    LocalResourceLoader,
    ResourceLoader,
)


@dataclass(frozen=True, repr=False)
class MultiPageDocument(BaseDataModel):
    """Multi-page document backed by a source path (local or remote URI).
    Holds no bytes, no open file/native handles as instance state — trivially
    picklable and safe to pass across process boundaries (e.g. torch
    DataLoader workers). Bytes are fetched and a pdfium handle opened fresh,
    on demand, each time a page is actually rendered."""

    source_path: str
    dpi: int = 200

    @classmethod
    def from_pdf(
        cls,
        source: str | Path,
        dpi: int = 200,
    ) -> MultiPageDocument:
        return cls(
            source_path=str(source),
            dpi=dpi,
        )

    @property
    def num_pages(self) -> int:
        import pypdfium2 as pdfium

        pdf_bytes = ResourceLoader.for_uri(self.source_path).load_bytes()
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            return len(pdf)
        finally:
            pdf.close()

    def get_page(self, page_number: int) -> SinglePageDocument:
        import pypdfium2 as pdfium

        pdf_bytes = ResourceLoader.for_uri(self.source_path).load_bytes()
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            assert 0 <= page_number < len(pdf), (
                f"Page number {page_number} out of range [0, {len(pdf) - 1}]"
            )
            page = pdf[page_number]
            try:
                bitmap = page.render(scale=self.dpi / 72)
                image = bitmap.to_pil()
            finally:
                page.close()
        finally:
            pdf.close()

        return SinglePageDocument(
            image=image,
            source_path=self.source_path,
            page_id=page_number,
        )

    def __iter__(self) -> Iterator[SinglePageDocument]:
        for i in range(self.num_pages):
            yield self.get_page(i)

    def to_dict(self) -> dict[str, Any]:
        return {"source_path": self.source_path, "dpi": self.dpi}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiPageDocument:
        return cls(source_path=data["source_path"], dpi=data.get("dpi", 200))


@dataclass(frozen=True, repr=False)
class SinglePageDocument(BaseDataModel):
    """Plain data container for a single page document."""

    image: PILImage.Image
    source_path: str | None = None
    page_id: int | None = None
    content: DocumentContent | None = None

    @classmethod
    def from_image(
        cls,
        source: str | Path | PILImage.Image,
        page_id: int | None = None,
        content: DocumentContent | None = None,
    ) -> SinglePageDocument:
        if isinstance(source, PILImage.Image):
            image = source
            source_path = None
        else:
            source_path = str(source)
            loader = ResourceLoader.for_uri(source_path)

            if isinstance(loader, LocalResourceLoader) and loader.byte_range is None:
                image = PILImage.open(loader.path)
                image.load()
            else:
                image = PILImage.open(BytesIO(loader.load_bytes()))
                image.load()

        return cls(
            image=image,
            source_path=source_path,
            page_id=page_id,
            content=content,
        )

    def to_dict(self) -> dict[str, Any]:
        """Path-backed pages serialize as a path reference (ArtifactStore's
        row-based storage); pages with no source_path (e.g. constructed
        directly from a PIL image) serialize as embedded bytes instead of
        raising -- `image` is always eagerly populated here, unlike
        `Image.content`, so there's no "not loaded yet" state to prefer."""
        content = self.content.to_dict() if self.content is not None else None
        if self.source_path is not None:
            return {
                "source_path": self.source_path,
                "page_id": self.page_id,
                "content": content,
            }
        from atria_core.types._utilities._image_encoding import _image_to_bytes

        return {
            "image_bytes": _image_to_bytes(self.image),
            "page_id": self.page_id,
            "content": content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SinglePageDocument:
        content = data.get("content")
        doc_content = (
            DocumentContent.from_dict(content) if content is not None else None
        )
        if "image_bytes" in data:
            from atria_core.types._utilities._image_encoding import _bytes_to_image

            return cls(
                image=_bytes_to_image(data["image_bytes"]),
                page_id=data.get("page_id"),
                content=doc_content,
            )
        return cls.from_image(
            source=data["source_path"],
            page_id=data.get("page_id"),
            content=doc_content,
        )
