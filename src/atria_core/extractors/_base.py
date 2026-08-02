from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from PIL.Image import Image as PILImage

from atria_core.types import DocumentContent, SinglePageDocumentInstance


class ContentExtractor(ABC):
    """Turns one document page's image content into structured
    `DocumentContent`. Multi-page fan-out isn't this class's concern --
    iterate a `MultiPageDocumentInstance` and call this per page."""

    @abstractmethod
    def _extract(self, image: PILImage) -> DocumentContent: ...

    def __call__(self, doc: SinglePageDocumentInstance) -> SinglePageDocumentInstance:
        loaded = doc.load()
        return replace(loaded, content=self._extract(loaded.require_content()))


class ContentExtractorConfig(ABC):
    @abstractmethod
    def build(self) -> ContentExtractor: ...
