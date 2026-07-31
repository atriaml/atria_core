from __future__ import annotations

import json

from atria_core.types._generic._bounding_box import BoundingBox


class TextElement:
    def __init__(
        self,
        text: str | None = None,
        bbox: BoundingBox | None = None,
        segment_bbox: BoundingBox | None = None,
        conf: float | None = None,
        angle: float | None = None,
    ) -> None:
        self.text = text
        self.bbox = bbox
        self.segment_bbox = segment_bbox
        self.conf = conf
        self.angle = angle

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": {
                "value": self.bbox.value,
                "mode": self.bbox.mode.value,
                "normalized": self.bbox.normalized,
            }
            if self.bbox
            else None,
            "segment_bbox": {
                "value": self.segment_bbox.value,
                "mode": self.segment_bbox.mode.value,
                "normalized": self.segment_bbox.normalized,
            }
            if self.segment_bbox
            else None,
            "conf": self.conf,
            "angle": self.angle,
        }

    def __repr__(self) -> str:
        return f"TextElement(text={self.text!r}, conf={self.conf}, angle={self.angle})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextElement):
            return NotImplemented
        return (
            self.text == other.text
            and self.bbox == other.bbox
            and self.segment_bbox == other.segment_bbox
            and self.conf == other.conf
            and self.angle == other.angle
        )


class DocumentContent:
    def __init__(
        self,
        text: str | None = None,
        text_elements: list[TextElement] | str | None = None,
    ) -> None:
        self.text_elements = self._parse_text_elements(text_elements)
        # derive text from elements if not provided
        if text is None and self.text_elements is not None:
            texts = [te.text for te in self.text_elements if te.text is not None]
            self.text = " ".join(texts) if texts else None
        else:
            self.text = text

    @staticmethod
    def _parse_text_elements(
        value: list[TextElement] | str | None,
    ) -> list[TextElement] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = json.loads(value)
        return [TextElement(**el) if isinstance(el, dict) else el for el in value]

    def serialize_text_elements(self) -> str | None:
        if self.text_elements is None:
            return None
        return json.dumps([te.to_dict() for te in self.text_elements])

    @property
    def text_list(self) -> list[str]:
        if self.text_elements is None:
            return []
        return [te.text for te in self.text_elements if te.text is not None]

    @property
    def bbox_list(self) -> list[BoundingBox]:
        if self.text_elements is None:
            return []
        return [te.bbox for te in self.text_elements if te.bbox is not None]

    @property
    def segment_bbox_list(self) -> list[BoundingBox]:
        if self.text_elements is None:
            return []
        return [
            te.segment_bbox for te in self.text_elements if te.segment_bbox is not None
        ]

    def __repr__(self) -> str:
        n = len(self.text_elements) if self.text_elements else 0
        return f"DocumentContent(text={self.text!r}, text_elements={n} elements)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentContent):
            return NotImplemented
        return self.text == other.text and self.text_elements == other.text_elements
