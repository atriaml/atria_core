from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._annotations import AnnotationType, LabelMap


@dataclass(repr=False)
class EntityLabelingAnnotation(BaseDataModel):
    type = AnnotationType.entity_labeling.value

    word_labels: list[int]
    label_map: LabelMap

    def __post_init__(self) -> None:
        if self.annotated_objects is not None:
            for obj in self.annotated_objects:
                if obj.label < 0 or obj.label >= len(self.label_map):
                    raise ValueError(
                        f"Invalid object label index {obj.label}. "
                        f"Label map contains only {len(self.label_map)} labels."
                    )

    @property
    def label_names(self) -> list[str]:
        return [self.label_map[label] for label in self.word_labels]

    def serialize_word_labels(self) -> str:
        return json.dumps(self.word_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "word_labels": self.word_labels,
            "label_map": self.label_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityLabelingAnnotation:
        return cls(word_labels=list(data["word_labels"]), label_map=data["label_map"])


entity = EntityLabelingAnnotation(word_labels=[1, 2, 3], label_map=["a", "b", "c"])
print(entity)
