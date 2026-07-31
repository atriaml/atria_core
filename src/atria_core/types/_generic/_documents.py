from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image as PILImage

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._doc_content import DocumentContent
from atria_core.types._utilities._url_fetchers import (
    LocalResourceLoader,
    ResourceLoader,
)

if TYPE_CHECKING:
    from atria_core.types._extractors._base import (
        ContentExtractor,
        ContentExtractorConfig,
    )


@dataclass(repr=False)
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


@dataclass(repr=False)
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

    def extract_content(
        self, extractor: ContentExtractor | ContentExtractorConfig
    ) -> SinglePageDocument:
        """Runs an extractor over this page and returns a new document with content attached."""
        from atria_core.types._extractors._base import ContentExtractorConfig

        if isinstance(extractor, ContentExtractorConfig):
            extractor = extractor.build()

        return extractor(self)

    def to_dict(self) -> dict[str, Any]:
        if self.source_path is None:
            raise ValueError(
                "SinglePageDocument must have a source_path before to_dict() "
                "-- materialize in-memory content to a file first (see ArtifactStore)."
            )
        return {
            "source_path": self.source_path,
            "page_id": self.page_id,
            "content": self.content.to_dict() if self.content is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SinglePageDocument:
        content = data.get("content")
        return cls.from_image(
            source=data["source_path"],
            page_id=data.get("page_id"),
            content=DocumentContent.from_dict(content) if content is not None else None,
        )
