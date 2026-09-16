from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from atria_core.types._data_instance._base import DataInstance
from atria_core.types._generic._doc_content import DocumentContent
from atria_core.types._generic._documents import PdfPage
from atria_core.types._generic._image import Image
from atria_core.types._utilities._url_fetchers import ResourceLoader


@dataclass(frozen=True, repr=False)
class DocumentInstance(DataInstance):
    """Abstract base -- only SinglePageDocumentInstance/MultiPageDocumentInstance
    are ever constructed. Shared only for isinstance checks and typing."""


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
            "metadata": self.metadata,
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
            metadata=data.get("metadata", {}),
            visual=visual_cls.from_dict(data["visual"]),
            content=DocumentContent.from_dict(content) if content is not None else None,
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )


@dataclass(frozen=True, repr=False)
class MultiPageDocumentInstance(DocumentInstance):
    pages: list[SinglePageDocumentInstance]

    @classmethod
    def from_pdf(
        cls, source_path: str | Path, sample_id: str | None = None, dpi: int = 200
    ) -> MultiPageDocumentInstance:
        import pypdfium2 as pdfium

        sample_id = sample_id if sample_id is not None else Path(source_path).name
        pdf_bytes = ResourceLoader.for_uri(str(source_path)).load_bytes()
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            num_pages = len(pdf)
        finally:
            pdf.close()
        return cls(
            sample_id=sample_id,
            pages=[
                SinglePageDocumentInstance.from_pdf(
                    source_path, page_id=page_number, dpi=dpi
                )
                for page_number in range(num_pages)
            ],
        )

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def get_page(self, page_number: int) -> SinglePageDocumentInstance:
        assert 0 <= page_number < self.num_pages, (
            f"Page number {page_number} out of range [0, {self.num_pages - 1}]"
        )
        return self.pages[page_number]

    def __iter__(self) -> Iterator[SinglePageDocumentInstance]:
        return iter(self.pages)

    def load(self) -> MultiPageDocumentInstance:
        return replace(self, pages=[page.load() for page in self.pages])

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metadata": self.metadata,
            "pages": [page.to_dict() for page in self.pages],
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiPageDocumentInstance:
        return cls(
            sample_id=data["sample_id"],
            metadata=data.get("metadata", {}),
            pages=[
                SinglePageDocumentInstance.from_dict(page) for page in data["pages"]
            ],
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )
