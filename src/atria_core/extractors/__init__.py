# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._base import ContentExtractor, ContentExtractorConfig
    from ._pdf_native import PdfNativeExtractor
    from ._tesseract import TesseractExtractor, TesseractExtractorConfig

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_base": ["ContentExtractor", "ContentExtractorConfig"],
        "_pdf_native": ["PdfNativeExtractor"],
        "_tesseract": ["TesseractExtractor", "TesseractExtractorConfig"],
    },
)
