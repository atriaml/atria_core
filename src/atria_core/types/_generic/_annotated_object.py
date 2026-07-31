from __future__ import annotations

from atria_core.types._generic._bounding_box import BoundingBox
from atria_core.types._generic._label import Label


class AnnotatedObject:
    def __init__(
        self,
        label: Label,
        bbox: BoundingBox,
        segmentation: list[list[float]] | None = None,
        iscrowd: bool = False,
    ) -> None:
        self.label = label
        self.bbox = bbox
        self.segmentation = segmentation
        self.iscrowd = iscrowd

    def __repr__(self) -> str:
        return (
            f"AnnotatedObject(label={self.label!r}, bbox={self.bbox!r}, "
            f"segmentation={self.segmentation!r}, iscrowd={self.iscrowd})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnotatedObject):
            return NotImplemented
        return (
            self.label == other.label
            and self.bbox == other.bbox
            and self.segmentation == other.segmentation
            and self.iscrowd == other.iscrowd
        )
