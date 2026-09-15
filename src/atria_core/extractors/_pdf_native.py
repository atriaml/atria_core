from __future__ import annotations

import io
from dataclasses import replace
from typing import Any

import numpy as np

from atria_core.logger import get_logger
from atria_core.types import (
    DocumentContent,
    ElementArray,
    MultiPageDocumentInstance,
    OCRLevel,
    PdfPage,
    ResourceLoader,
    SinglePageDocumentInstance,
)

logger = get_logger(__name__)


class PdfNativeExtractor:
    """Reads a PDF's embedded text layer directly via pdfplumber, instead of
    OCR-ing a rasterized page. Doesn't fit ContentExtractor's per-image
    `_extract()` contract -- it needs the PDF's text layer, not pixels -- so
    it's its own class, called on a whole MultiPageDocumentInstance (mirroring
    ContentExtractor.__call__'s "operates on documents" shape).

    """

    def __init__(self, x_tolerance: float = 1.0, y_tolerance: float = 1.0) -> None:
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance

    def __call__(
        self, document: MultiPageDocumentInstance
    ) -> list[SinglePageDocumentInstance]:
        import pdfplumber

        first_page = document.get_page(0).visual
        assert isinstance(first_page, PdfPage) and first_page.file_path is not None
        pdf_bytes = ResourceLoader.for_uri(first_page.file_path).load_bytes()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            contents = [self._extract_page(page) for page in pdf.pages]

        return [
            replace(document.get_page(i), content=content)
            for i, content in enumerate(contents)
        ]

    def _extract_page(self, page: Any) -> DocumentContent:
        width, height = float(page.width), float(page.height)
        words = [
            w
            for w in page.extract_words(
                x_tolerance=self.x_tolerance, y_tolerance=self.y_tolerance
            )
            if w["text"].strip()
        ]
        try:
            lines = page.extract_text_lines(
                x_tolerance=self.x_tolerance, y_tolerance=self.y_tolerance
            )
        except Exception:  # older pdfplumber, or a page it can't line-split
            logger.debug("extract_text_lines failed; words only", exc_info=True)
            lines = []

        ids: list[int] = [0]
        parent_ids: list[int] = [-1]
        levels: list[int] = [OCRLevel.page.value]
        bboxes: list[tuple[float, float, float, float]] = [(0.0, 0.0, 1.0, 1.0)]
        texts: list[str] = [""]
        confs: list[float] = [float("nan")]
        page_id = 0

        line_ids: list[int] = []
        for line in lines:
            line_id = len(ids)
            ids.append(line_id)
            parent_ids.append(page_id)
            levels.append(OCRLevel.line.value)
            bboxes.append(self._normalize(line, width, height))
            texts.append(line.get("text", ""))
            confs.append(float("nan"))
            line_ids.append(line_id)

        for word in words:
            parent_id = self._line_for_word(word, lines, line_ids)
            ids.append(len(ids))
            parent_ids.append(parent_id if parent_id is not None else page_id)
            levels.append(OCRLevel.word.value)
            bboxes.append(self._normalize(word, width, height))
            texts.append(word["text"].strip())
            confs.append(100.0)  # native text isn't a prediction

        elements = ElementArray(
            ids=np.array(ids),
            parent_ids=np.array(parent_ids),
            levels=np.array(levels),
            bboxes=np.asarray(bboxes, dtype=np.float64),
            normalized=True,
            texts=np.asarray(texts, dtype=object),
            confs=np.asarray(confs, dtype=np.float64),
        )
        elements.validate_hierarchy()
        return DocumentContent(elements=elements)

    @staticmethod
    def _normalize(
        box: dict[str, float], width: float, height: float
    ) -> tuple[float, float, float, float]:
        return (
            box["x0"] / width,
            box["top"] / height,
            box["x1"] / width,
            box["bottom"] / height,
        )

    @staticmethod
    def _line_for_word(
        word: dict[str, float], lines: list[dict[str, float]], line_ids: list[int]
    ) -> int | None:
        """The line whose box contains the word's center, or None if no line
        claims it (the word then hangs off the page root instead)."""
        cx = (word["x0"] + word["x1"]) / 2
        cy = (word["top"] + word["bottom"]) / 2
        for line, line_id in zip(lines, line_ids, strict=True):
            if line["x0"] <= cx <= line["x1"] and line["top"] <= cy <= line["bottom"]:
                return line_id
        return None
