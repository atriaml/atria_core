from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

from atria_core.types._extractors._base import ContentExtractor, ContentExtractorConfig
from atria_core.types._generic._doc_content import DocumentContent
from atria_core.types._utilities._url_fetchers import (
    LocalResourceLoader,
    ResourceLoader,
)


class MultiPageDocument:
    """Multi-page document backed by a source path (local or remote URI).
    Holds no bytes, no open file/native handles as instance state — trivially
    picklable and safe to pass across process boundaries (e.g. torch
    DataLoader workers). Bytes are fetched and a pdfium handle opened fresh,
    on demand, each time a page is actually rendered."""

    def __init__(
        self,
        source_path: str,
        dpi: int = 200,
    ) -> None:
        self.source_path = source_path
        self.dpi = dpi

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
            sample_id=f"{self.sample_id}_page_{page_number}",
            image=image,
            source_path=self.source_path,
            page_id=page_number,
        )

    def __iter__(self):
        for i in range(self.num_pages):
            yield self.get_page(i)

    def __repr__(self) -> str:
        return f"MultiPageDocument(sample_id={self.sample_id!r}, source={self.source_path!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiPageDocument):
            return NotImplemented
        return super().__eq__(other) and self.source_path == other.source_path


class SinglePageDocument:
    """Plain data container for a single page document."""

    def __init__(
        self,
        image: PILImage.Image,
        source_path: str | None = None,
        page_id: int | None = None,
        content: DocumentContent | None = None,
    ) -> None:
        self.image = image
        self.source_path = source_path
        self.page_id = page_id
        self.content = content

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
        if isinstance(extractor, ContentExtractorConfig):
            extractor = extractor.build()

        return SinglePageDocument(
            image=self.image,
            source_path=self.source_path,
            page_id=self.page_id,
            content=extractor(self.image),
        )

    def __repr__(self) -> str:
        return (
            f"SinglePageDocument(source={self.source_path!r}, "
            f"page_id={self.page_id}, image={self.image.size}, content={self.content is not None})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SinglePageDocument):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.image == other.image
            and self.source_path == other.source_path
            and self.page_id == other.page_id
            and self.content == other.content
        )
