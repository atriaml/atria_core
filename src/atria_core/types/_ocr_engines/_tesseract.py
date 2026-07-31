from typing import Any, Literal

import numpy as np
import pytesseract
from PIL.Image import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field

from atria_core.logger import get_logger
from atria_core.types import Image, TextElement
from atria_core.types._ocr_engines._base import BaseOCREngine

logger = get_logger(__name__)


class TesseractOCREngineConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        frozen=True,
    )

    type: Literal["tesseract"] = "tesseract"
    lang: str = Field(default="eng", description="Language code(s) for OCR")
    psm: int | None = Field(
        default=3,
        description="Page segmentation mode (0-13). Default 3: Fully automatic page segmentation, but no OSD",
    )
    oem: int | None = Field(
        default=3,
        description="OCR Engine mode (0-3). Default 3: LSTM neural net mode only (best accuracy)",
    )

    def build(self) -> "TesseractOCREngine":
        return TesseractOCREngine(config=self)


class TesseractOCREngine(BaseOCREngine):
    def __init__(self, config: TesseractOCREngineConfig):
        self.config = config

    def _build_config_string(self) -> str:
        config_parts = []
        if self.config.psm is not None:
            config_parts.append(f"--psm {self.config.psm}")
        if self.config.oem is not None:
            config_parts.append(f"--oem {self.config.oem}")
        return " ".join(config_parts) if config_parts else ""

    def _get_tesseract_data(self, image: np.ndarray) -> dict[str, Any]:
        config = self._build_config_string()
        return pytesseract.image_to_data(
            image,
            lang=self.config.lang,
            config=config,
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
        segment_dict = {}
        for word_info in word_infos:
            segment_id = word_info["segment_id"]
            if segment_id not in segment_dict:
                segment_dict[segment_id] = []
            segment_dict[segment_id].append(word_info)
        return segment_dict

    def _calculate_line_bbox(self, line_words: list[dict[str, Any]]) -> tuple:
        line_left = min(w["bbox"][0] for w in line_words)
        line_top = min(w["bbox"][1] for w in line_words)
        line_right = max(w["bbox"][2] for w in line_words)
        line_bottom = max(w["bbox"][3] for w in line_words)
        return (line_left, line_top, line_right, line_bottom)

    def _create_text_element(
        self,
        word_info: dict[str, Any],
        segment_bbox: tuple,
        image_width: int,
        image_height: int,
    ) -> TextElement:
        from atria_core.types import BoundingBox, BoundingBoxMode

        return TextElement(
            text=word_info["text"],
            bbox=BoundingBox(
                value=[
                    word_info["bbox"][0],
                    word_info["bbox"][1],
                    word_info["bbox"][2],
                    word_info["bbox"][3],
                ],
                mode=BoundingBoxMode.XYXY,
            ).ops.normalize(image_width, image_height),
            conf=word_info["conf"],
            segment_bbox=BoundingBox(
                value=[
                    segment_bbox[0],
                    segment_bbox[1],
                    segment_bbox[2],
                    segment_bbox[3],
                ],
                mode=BoundingBoxMode.XYXY,
            ).ops.normalize(image_width, image_height),
        )

    def _preprocess_image(self, image: PILImage) -> np.ndarray:
        return np.array(image.convert("L"))

    def extract_text_elements(self, image: Image) -> list[TextElement]:
        # preprocess image
        preprocessed_image = self._preprocess_image(image.content)

        # extract tesseract data
        data = self._get_tesseract_data(preprocessed_image)

        # extract word infos
        word_infos = []
        for i in range(len(data["text"])):
            word_info = self._extract_word_info(data, i)
            if word_info is not None:
                word_infos.append(word_info)

        # group words by line id (this actually includes (block, paragraph, line) tuple)
        segments_dict = self._group_words_by_segment(word_infos)

        # create text elements
        text_elements = []
        for word_info in word_infos:
            segment_id = word_info["segment_id"]
            segment_words = segments_dict[segment_id]
            segment_bbox = self._calculate_line_bbox(segment_words)
            text_element = self._create_text_element(
                word_info,
                segment_bbox,
                image_width=image.width,
                image_height=image.height,
            )
            text_elements.append(text_element)

        return text_elements
