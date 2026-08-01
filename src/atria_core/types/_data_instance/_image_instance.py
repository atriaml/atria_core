from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._generic._image import Image


@dataclass(frozen=True, repr=False)
class ImageInstance(BaseDataInstance):
    image: Image

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image": self.image.to_dict(),
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageInstance:
        return cls(
            sample_id=data["sample_id"],
            image=Image.from_dict(data["image"]),
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )
