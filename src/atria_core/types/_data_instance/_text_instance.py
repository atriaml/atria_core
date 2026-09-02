from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._data_instance._base import DataInstance, Metadata


@dataclass(frozen=True, repr=False)
class TextInstance(DataInstance):
    """Plain text with no visual/page representation at all -- for
    text-only datasets (e.g. SQuAD)."""

    text: str

    def load(self) -> TextInstance:
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metadata": self.metadata.to_dict(),
            "text": self.text,
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextInstance:
        return cls(
            sample_id=data["sample_id"],
            metadata=Metadata.from_dict(data.get("metadata", {})),
            text=data["text"],
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )
