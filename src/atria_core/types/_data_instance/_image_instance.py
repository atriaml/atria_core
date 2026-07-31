from __future__ import annotations

from atria_core.types._data_instance._base import (
    BaseDataInstance,
)
from atria_core.types._generic._annotations import Annotation
from atria_core.types._generic._image import Image


class ImageInstance(BaseDataInstance):
    def __init__(
        self,
        sample_id: str,
        image: Image,
        annotations: list[Annotation] | None = None,
    ) -> None:
        super().__init__(sample_id=sample_id, annotations=annotations)
        self.image = image

    def __repr__(self) -> str:
        n = len(self.annotations) if self.annotations else 0
        return (
            f"ImageInstance(sample_id={self.sample_id!r}, "
            f"image={self.image!r}, annotations={n})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageInstance):
            return NotImplemented
        return super().__eq__(other) and self.image == other.image
