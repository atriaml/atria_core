from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._elements import ElementArray, OCRLevel


@dataclass(repr=False)
class DocumentContent(BaseDataModel):
    """OCR/text-layer output for one page. `elements` holds the full
    page/block/paragraph/line/word hierarchy as a structure-of-arrays (see
    ElementArray) -- word bboxes and their line/segment boxes both live
    there, with no separate stored field to drift out of sync."""

    text: str | None = None
    elements: ElementArray | None = None

    def __post_init__(self) -> None:
        if self.text is None and self.elements is not None:
            joined = self.elements.joined_text()
            self.text = joined or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "elements": self.elements.to_dict() if self.elements is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentContent:
        elements = data.get("elements")
        return cls(
            text=data.get("text"),
            elements=ElementArray.from_dict(elements) if elements is not None else None,
        )
