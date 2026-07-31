from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytesseract
from PIL.Image import Image as PILImage

from atria_core.logger import get_logger
from atria_core.types._extractors._base import ContentExtractor, ContentExtractorConfig
from atria_core.types._generic._bounding_box import BoundingBox, BoundingBoxMode
from atria_core.types._generic._doc_content import DocumentContent, TextElement

logger = get_logger(__name__)


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
    def __init__(self, config: TesseractExtractorConfig):
        self.config = config

    def __call__(self, image: PILImage) -> DocumentContent:
        preprocessed = self._preprocess_image(image)
        data = self._get_tesseract_data(preprocessed)

        word_infos = []
        for i in range(len(data["text"])):
            word_info = self._extract_word_info(data, i)
            if word_info is not None:
                word_infos.append(word_info)

        segments_dict = self._group_words_by_segment(word_infos)

        text_elements = []
        for word_info in word_infos:
            segment_words = segments_dict[word_info["segment_id"]]
            segment_bbox = self._calculate_line_bbox(segment_words)
            text_elements.append(
                self._create_text_element(
                    word_info, segment_bbox, image.width, image.height
                )
            )

        return DocumentContent(text_elements=text_elements)

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

    def _extract_word_info(
        self, data: dict[str, Any], index: int
    ) -> dict[str, Any] | None:
        text = data["text"][index].strip()
        if not text:
            return None

        bbox = (
            data["left"][index],
            data["top"][index],
            data["left"][index] + data["width"][index],
            data["top"][index] + data["height"][index],
        )
        conf = float(data["conf"][index]) if data["conf"][index] != "-1" else None
        segment_id = (
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index],
        )

        return {"text": text, "bbox": bbox, "conf": conf, "segment_id": segment_id}

    def _group_words_by_segment(
        self, word_infos: list[dict[str, Any]]
    ) -> dict[tuple, list[dict[str, Any]]]:
        segment_dict: dict[tuple, list[dict[str, Any]]] = {}
        for word_info in word_infos:
            segment_dict.setdefault(word_info["segment_id"], []).append(word_info)
        return segment_dict

    def _calculate_line_bbox(self, line_words: list[dict[str, Any]]) -> tuple:
        return (
            min(w["bbox"][0] for w in line_words),
            min(w["bbox"][1] for w in line_words),
            max(w["bbox"][2] for w in line_words),
            max(w["bbox"][3] for w in line_words),
        )

    def _create_text_element(
        self,
        word_info: dict[str, Any],
        segment_bbox: tuple,
        image_width: int,
        image_height: int,
    ) -> TextElement:
        return TextElement(
            text=word_info["text"],
            bbox=BoundingBox(
                value=list(word_info["bbox"]), mode=BoundingBoxMode.XYXY
            ).normalize(image_width, image_height),
            conf=word_info["conf"],
            segment_bbox=BoundingBox(
                value=list(segment_bbox), mode=BoundingBoxMode.XYXY
            ).normalize(image_width, image_height),
        )

    def _preprocess_image(self, image: PILImage) -> np.ndarray:
        return np.array(image.convert("L"))
