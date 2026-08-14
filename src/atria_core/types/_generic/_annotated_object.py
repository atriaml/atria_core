from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._arrays import FloatArray
from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._bounding_box import (
    as_bbox_array,
    as_segmentation_array,
    check_bbox_array,
    check_segmentation_array,
)


@dataclass(frozen=True, repr=False, eq=False)
class AnnotatedObject(BaseDataModel):
    """One detected/annotated object. A human-readable, easy-to-construct
    counterpart to ObjectDetectionAnnotation's array-backed storage -- build
    a list of these and pass it to ObjectDetectionAnnotation.from_objects()."""

    label_value: int
    bbox: FloatArray
    label_name: str | None = None
    segmentation: FloatArray | None = None
    iscrowd: bool = False

    def __post_init__(self) -> None:
        check_bbox_array(self.bbox)
        if self.segmentation is not None:
            check_segmentation_array(self.segmentation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_value": self.label_value,
            "label_name": self.label_name,
            "bbox": self.bbox.tolist(),
            "segmentation": self.segmentation.tolist()
            if self.segmentation is not None
            else None,
            "iscrowd": self.iscrowd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotatedObject:
        segmentation = data.get("segmentation")
        return cls(
            label_value=data["label_value"],
            label_name=data["label_name"],
            bbox=as_bbox_array(data["bbox"]),
            segmentation=as_segmentation_array(segmentation)
            if segmentation is not None
            else None,
            iscrowd=data.get("iscrowd", False),
        )
