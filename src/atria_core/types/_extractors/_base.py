from __future__ import annotations

from abc import ABC, abstractmethod

from atria_core.types._data_instance._document import (
    MultiPageDocument,
    SinglePageDocument,
)
from PIL.Image import Image as PILImage

from atria_core.types._generic._doc_content import DocumentContent


class ContentExtractor(ABC):
    @abstractmethod
    def _extract(self, image: PILImage) -> DocumentContent: ...

    def __call__(self, doc: SinglePageDocument | MultiPageDocument):
        if isinstance(doc, MultiPageDocument):
            return [self(page) for page in doc]

        return SinglePageDocument(
            sample_id=doc.sample_id,
            image=doc.image,
            source_path=doc.source_path,
            page_id=doc.page_id,
            content=self._extract(doc.image),
            annotations=doc.annotations,
        )


class ContentExtractorConfig(ABC):
    @abstractmethod
    def build(self) -> ContentExtractor: ...
