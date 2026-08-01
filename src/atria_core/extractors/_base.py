from __future__ import annotations

from abc import ABC, abstractmethod
from typing import overload

from PIL.Image import Image as PILImage

from atria_core.types import DocumentContent, MultiPageDocument, SinglePageDocument


class ContentExtractor(ABC):
    """Turns a document's image content into structured `DocumentContent`.

    Conforms to the `Stage` convention (see `atria_core.types._utilities._stage`):
    a typed, composable callable. Subclasses implement `_extract` for a single
    page's image; `__call__` owns the page-recursion policy for multi-page
    documents. Documents stay pure value objects here — no instance-level
    concerns (`sample_id`, `annotations`) are threaded through.
    """

    @abstractmethod
    def _extract(self, image: PILImage) -> DocumentContent: ...

    @overload
    def __call__(self, doc: SinglePageDocument) -> SinglePageDocument: ...
    @overload
    def __call__(self, doc: MultiPageDocument) -> list[SinglePageDocument]: ...

    def __call__(
        self, doc: SinglePageDocument | MultiPageDocument
    ) -> SinglePageDocument | list[SinglePageDocument]:
        if isinstance(doc, MultiPageDocument):
            return [self(page) for page in doc]

        return SinglePageDocument(
            image=doc.image,
            source_path=doc.source_path,
            page_id=doc.page_id,
            content=self._extract(doc.image),
        )


class ContentExtractorConfig(ABC):
    @abstractmethod
    def build(self) -> ContentExtractor: ...
