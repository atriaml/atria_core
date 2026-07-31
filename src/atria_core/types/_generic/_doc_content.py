from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._bounding_box import BoundingBoxMode, as_bbox_array


@dataclass(repr=False, eq=False)
class TextElement(BaseDataModel):
    text: str | None = None
    bbox: np.ndarray | None = None
    segment_bbox: np.ndarray | None = None
    conf: float | None = None
    angle: float | None = None

    def __post_init__(self) -> None:
        if self.bbox is not None:
            self.bbox = as_bbox_array(self.bbox)
        if self.segment_bbox is not None:
            self.segment_bbox = as_bbox_array(self.segment_bbox)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "segment_bbox": self.segment_bbox.tolist()
            if self.segment_bbox is not None
            else None,
            "conf": self.conf,
            "angle": self.angle,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextElement:
        return cls(
            text=data.get("text"),
            bbox=data.get("bbox"),
            segment_bbox=data.get("segment_bbox"),
            conf=data.get("conf"),
            angle=data.get("angle"),
        )


@dataclass(repr=False)
class DocumentContent(BaseDataModel):
    text: str | None = None
    text_elements: list[TextElement] | None = None
    bbox_mode: BoundingBoxMode = BoundingBoxMode.XYXY
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.text is None and self.text_elements is not None:
            texts = [te.text for te in self.text_elements if te.text is not None]
            self.text = " ".join(texts) if texts else None

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
    def bbox_list(self) -> list[np.ndarray]:
        if self.text_elements is None:
            return []
        return [te.bbox for te in self.text_elements if te.bbox is not None]

    @property
    def segment_bbox_list(self) -> list[np.ndarray]:
        if self.text_elements is None:
            return []
        return [
            te.segment_bbox for te in self.text_elements if te.segment_bbox is not None
        ]

    # -------------------------------------
    # Generic batch-transform protocol (see _transforms/_bounding_box.py)
    # -------------------------------------
    def box_batches(self) -> dict[str, tuple[np.ndarray, list[int]] | None]:
        """Named batches of stacked bboxes, each paired with the text_elements
        indices they came from (some elements may have no bbox/segment_bbox)."""
        getters: list[tuple[str, Callable[[TextElement], np.ndarray | None]]] = [
            ("bbox", lambda te: te.bbox),
            ("segment_bbox", lambda te: te.segment_bbox),
        ]
        batches: dict[str, tuple[np.ndarray, list[int]] | None] = {}
        for name, getter in getters:
            if self.text_elements is None:
                batches[name] = None
                continue
            present = [
                (i, box)
                for i, te in enumerate(self.text_elements)
                if (box := getter(te)) is not None
            ]
            batches[name] = (
                (np.stack([box for _, box in present]), [i for i, _ in present])
                if present
                else None
            )
        return batches

    def with_box_batches(
        self,
        batches: dict[str, tuple[np.ndarray, list[int]]],
        *,
        normalized: bool,
        mode: BoundingBoxMode,
    ) -> DocumentContent:
        if self.text_elements is None:
            return replace(self, normalized=normalized, bbox_mode=mode)

        text_elements = list(self.text_elements)
        for name in ("bbox", "segment_bbox"):
            if name not in batches:
                continue
            new_values, indices = batches[name]
            for row, i in zip(new_values, indices, strict=True):
                text_elements[i] = replace(text_elements[i], **{name: row})

        return replace(
            self, text_elements=text_elements, normalized=normalized, bbox_mode=mode
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_elements": [te.to_dict() for te in self.text_elements]
            if self.text_elements is not None
            else None,
            "bbox_mode": self.bbox_mode.value,
            "normalized": self.normalized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentContent:
        text_elements = data.get("text_elements")
        return cls(
            text=data.get("text"),
            text_elements=[TextElement.from_dict(te) for te in text_elements]
            if text_elements is not None
            else None,
            bbox_mode=BoundingBoxMode(data.get("bbox_mode", BoundingBoxMode.XYXY.value)),
            normalized=data.get("normalized", False),
        )
