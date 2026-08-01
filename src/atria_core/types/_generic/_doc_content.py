from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._elements import ElementArray


@dataclass(frozen=True, repr=False)
class DocumentContent(BaseDataModel):
    """OCR/text-layer output for one page. `elements` holds the full
    page/block/paragraph/line/word hierarchy as a structure-of-arrays (see
    ElementArray) -- word bboxes and their line/segment boxes both live
    there, with no separate stored field to drift out of sync."""

    __repr_fields__ = {"text", "elements"}

    elements: ElementArray | None = None
    #: Explicit override for `text`, if the caller has one. Private: read
    #: `text` instead, which falls back to `elements.joined_text()`.
    _text: str | None = field(default=None, kw_only=True)

    @property
    def text(self) -> str | None:
        if self._text is not None:
            return self._text
        if self.elements is not None:
            return self.elements.joined_text() or None
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "elements": self.elements.to_dict() if self.elements is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentContent:
        elements = data.get("elements")
        return cls(
            _text=data.get("text"),
            elements=ElementArray.from_dict(elements) if elements is not None else None,
        )
