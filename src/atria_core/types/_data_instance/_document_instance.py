from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._generic._doc_content import DocumentContent
from atria_core.types._generic._documents import PdfPage
from atria_core.types._generic._image import Image
from atria_core.types._utilities._url_fetchers import ResourceLoader


@dataclass(frozen=True, repr=False)
class DocumentInstance(BaseDataInstance):
    """Abstract base -- only SinglePageDocumentInstance/MultiPageDocumentInstance
    are ever constructed. Shared only for isinstance checks and typing
    (e.g. DocumentDataset.__data_model__)."""


@dataclass(frozen=True, repr=False)
class SinglePageDocumentInstance(DocumentInstance):
    """One document page -- a visual (Image or PdfPage, loaded lazily by
    whichever it is) plus optional extracted content. Loading/serialization
    is entirely delegated to `visual`."""

    visual: Image | PdfPage
    content: DocumentContent | None = None

    @classmethod
    def from_image(
        cls,
        source: str | Path | PILImage.Image,
        sample_id: str | None = None,
        content: DocumentContent | None = None,
    ) -> SinglePageDocumentInstance:
        if sample_id is None:
            assert not isinstance(source, PILImage.Image), (
                "sample_id must be given explicitly when source is a raw PIL image "
                "-- there's no path to derive it from."
            )
            sample_id = Path(source).name
        return cls(
            sample_id=sample_id, visual=Image.from_source(source), content=content
        )

    @classmethod
    def from_pdf(
        cls,
        source: str | Path,
        page_id: int,
        dpi: int = 200,
        content: DocumentContent | None = None,
    ) -> SinglePageDocumentInstance:
        return cls(
            sample_id=f"{Path(source).name}#{page_id}",
            visual=PdfPage(file_path=str(source), page_id=page_id, dpi=dpi),
            content=content,
        )

    @property
    def page_id(self) -> int | None:
        return self.visual.page_id if isinstance(self.visual, PdfPage) else None

    def load(self) -> SinglePageDocumentInstance:
        return replace(self, visual=self.visual.load())

    def require_content(self) -> PILImage.Image:
        return self.visual.require_content()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "visual_type": "pdf_page" if isinstance(self.visual, PdfPage) else "image",
            "visual": self.visual.to_dict(),
            "content": self.content.to_dict() if self.content is not None else None,
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SinglePageDocumentInstance:
        visual_cls: type[Image | PdfPage] = (
            PdfPage if data.get("visual_type") == "pdf_page" else Image
        )
        content = data.get("content")
        return cls(
            sample_id=data["sample_id"],
            visual=visual_cls.from_dict(data["visual"]),
            content=DocumentContent.from_dict(content) if content is not None else None,
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )


@dataclass(frozen=True, repr=False)
class MultiPageDocumentInstance(DocumentInstance):
    """Multi-page document backed by a source path (local or remote URI).
    Holds no bytes, no open file/native handles as instance state -- trivially
    picklable and safe to pass across process boundaries (e.g. torch
    DataLoader workers). Bytes are fetched and a pdfium handle opened fresh,
    on demand, each time a page is actually rendered."""

    source_path: str
    dpi: int = 200

    @property
    def num_pages(self) -> int:
        import pypdfium2 as pdfium

        pdf_bytes = ResourceLoader.for_uri(self.source_path).load_bytes()
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            return len(pdf)
        finally:
            pdf.close()

    def get_page(self, page_number: int) -> SinglePageDocumentInstance:
        assert 0 <= page_number < self.num_pages, (
            f"Page number {page_number} out of range [0, {self.num_pages - 1}]"
        )
        return SinglePageDocumentInstance(
            sample_id=f"{self.sample_id}#{page_number}",
            visual=PdfPage(
                file_path=self.source_path, page_id=page_number, dpi=self.dpi
            ),
            _annotations=dict(self._annotations),
        )

    def __iter__(self) -> Iterator[SinglePageDocumentInstance]:
        for i in range(self.num_pages):
            yield self.get_page(i)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_path": self.source_path,
            "dpi": self.dpi,
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiPageDocumentInstance:
        return cls(
            sample_id=data["sample_id"],
            source_path=data["source_path"],
            dpi=data.get("dpi", 200),
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )
