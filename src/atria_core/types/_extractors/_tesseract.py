from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytesseract
from PIL.Image import Image as PILImage

from atria_core.logger import get_logger
from atria_core.types._extractors._base import ContentExtractor, ContentExtractorConfig
from atria_core.types._generic._doc_content import DocumentContent
from atria_core.types._generic._elements import ElementArray, OCRLevel

logger = get_logger(__name__)

# pytesseract's level column: 1=page, 2=block, 3=paragraph, 4=line, 5=word.
_TESSERACT_LEVELS = {
    1: OCRLevel.page,
    2: OCRLevel.block,
    3: OCRLevel.paragraph,
    4: OCRLevel.line,
    5: OCRLevel.word,
}


class TesseractExtractorConfig(ContentExtractorConfig):
    type: Literal["tesseract"] = "tesseract"

    def __init__(
        self, lang: str = "eng", psm: int | None = 3, oem: int | None = 3
    ) -> None:
        self.lang = lang
        self.psm = psm
        self.oem = oem

    def build(self) -> TesseractExtractor:
        return TesseractExtractor(config=self)

    def __repr__(self) -> str:
        return f"TesseractExtractorConfig(lang={self.lang!r}, psm={self.psm}, oem={self.oem})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TesseractExtractorConfig):
            return NotImplemented
        return (
            self.lang == other.lang and self.psm == other.psm and self.oem == other.oem
        )


class TesseractExtractor(ContentExtractor):
    """Builds the real page->block->paragraph->line->word hierarchy from
    tesseract's own `level` column, instead of a flat word list with a
    reconstructed segment box. A word's line box is ElementArray.segment_bboxes()
    -- gathered from its parent, not stored redundantly."""

    def __init__(self, config: TesseractExtractorConfig):
        self.config = config

    def _extract(self, image: PILImage) -> DocumentContent:
        data = self._get_tesseract_data(self._preprocess_image(image))

        ids: list[int] = []
        parent_ids: list[int] = []
        levels: list[int] = []
        bboxes: list[tuple[float, float, float, float]] = []
        texts: list[str] = []
        confs: list[float] = []

        # Tesseract emits rows in document order, so the parent of a row at
        # level L is always the most recent row seen at level L - 1.
        last_id_at_level: dict[OCRLevel, int] = {}
        for i in range(len(data["text"])):
            level = _TESSERACT_LEVELS.get(int(data["level"][i]))
            if level is None:
                continue

            text = (data["text"][i] or "").strip()
            if level is OCRLevel.word and not text:
                continue  # tesseract emits blank word rows for spacing

            left, top = data["left"][i], data["top"][i]
            bbox = (
                left / image.width,
                top / image.height,
                (left + data["width"][i]) / image.width,
                (top + data["height"][i]) / image.height,
            )
            parent_level = OCRLevel(level.value - 1) if level != OCRLevel.page else None
            parent_id = last_id_at_level.get(parent_level, -1) if parent_level else -1
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else float("nan")

            element_id = len(ids)
            ids.append(element_id)
            parent_ids.append(parent_id)
            levels.append(level.value)
            bboxes.append(bbox)
            texts.append(text)
            confs.append(conf)
            last_id_at_level[level] = element_id

        if not ids:
            return DocumentContent(elements=None)

        elements = ElementArray(
            ids=np.array(ids),
            parent_ids=np.array(parent_ids),
            levels=np.array(levels),
            bboxes=np.asarray(bboxes, dtype=np.float64),
            texts=np.asarray(texts, dtype=object),
            confs=np.asarray(confs, dtype=np.float64),
        )
        return DocumentContent(elements=elements)

    def _build_config_string(self) -> str:
        parts = []
        if self.config.psm is not None:
            parts.append(f"--psm {self.config.psm}")
        if self.config.oem is not None:
            parts.append(f"--oem {self.config.oem}")
        return " ".join(parts)

    def _get_tesseract_data(self, image: np.ndarray) -> dict[str, Any]:
        return pytesseract.image_to_data(
            image,
            lang=self.config.lang,
            config=self._build_config_string(),
            output_type=pytesseract.Output.DICT,
        )

    def _preprocess_image(self, image: PILImage) -> np.ndarray:
        return np.array(image.convert("L"))
