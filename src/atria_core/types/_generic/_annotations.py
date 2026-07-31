from __future__ import annotations

import enum
import json
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._annotated_object import AnnotatedObject
from atria_core.types._generic._bounding_box import BoundingBoxMode
from atria_core.types._generic._qa_pair import QAPair


class AnnotationType(str, enum.Enum):
    classification = "classification"
    entity_labeling = "entity_labeling"
    question_answering = "question_answering"
    object_detection = "object_detection"
    layout_analysis = "layout_analysis"


@dataclass(repr=False)
class ClassificationAnnotation(BaseDataModel):
    type = AnnotationType.classification.value

    label: int
    label_map: list[str]

    def __post_init__(self) -> None:
        if self.label < 0 or self.label >= len(self.label_map):
            raise ValueError(
                f"Invalid label index {self.label}. "
                f"Label map contains only {len(self.label_map)} labels."
            )

    @property
    def label_name(self) -> str:
        return self.label_map[self.label]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "label": self.label, "label_map": self.label_map}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationAnnotation:
        return cls(label=data["label"], label_map=data["label_map"])


@dataclass(repr=False)
class EntityLabelingAnnotation(BaseDataModel):
    type = AnnotationType.entity_labeling.value

    word_labels: list[int]
    label_map: list[str]

    def __post_init__(self) -> None:
        if self.word_labels:
            max_label = max(self.word_labels)
            if max_label >= len(self.label_map):
                raise ValueError(
                    f"Invalid word label index {max_label}. "
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


@dataclass(repr=False)
class QuestionAnsweringAnnotation(BaseDataModel):
    type = AnnotationType.question_answering.value

    qa_pairs: list[QAPair]

    def serialize_qa_pairs(self) -> str:
        return json.dumps([q.to_dict() for q in self.qa_pairs])

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "qa_pairs": [q.to_dict() for q in self.qa_pairs]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionAnsweringAnnotation:
        return cls(qa_pairs=[QAPair.from_dict(q) for q in data["qa_pairs"]])


@dataclass(repr=False)
class ObjectDetectionAnnotation(BaseDataModel):
    type = AnnotationType.object_detection.value

    label_map: list[str]
    annotated_objects: list[AnnotatedObject] | None = None
    bbox_mode: BoundingBoxMode = BoundingBoxMode.XYXY
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.annotated_objects is not None:
            for obj in self.annotated_objects:
                if obj.label < 0 or obj.label >= len(self.label_map):
                    raise ValueError(
                        f"Invalid object label index {obj.label}. "
                        f"Label map contains only {len(self.label_map)} labels."
                    )

    def serialize_annotated_objects(self) -> str | None:
        if self.annotated_objects is None:
            return None
        return json.dumps([o.to_dict() for o in self.annotated_objects])

    # -------------------------------------
    # Generic batch-transform protocol (see _transforms/_bounding_box.py)
    # -------------------------------------
    def box_batches(self) -> dict[str, tuple[np.ndarray, list[int]] | None]:
        if not self.annotated_objects:
            return {"bbox": None}
        indices = list(range(len(self.annotated_objects)))
        return {
            "bbox": (
                np.stack([obj.bbox for obj in self.annotated_objects]),
                indices,
            )
        }

    def with_box_batches(
        self,
        batches: dict[str, tuple[np.ndarray, list[int]]],
        *,
        normalized: bool,
        mode: BoundingBoxMode,
    ) -> ObjectDetectionAnnotation:
        if self.annotated_objects is None or "bbox" not in batches:
            return replace(self, normalized=normalized, bbox_mode=mode)

        new_values, indices = batches["bbox"]
        annotated_objects = list(self.annotated_objects)
        for row, i in zip(new_values, indices, strict=True):
            annotated_objects[i] = replace(annotated_objects[i], bbox=row)

        return replace(
            self,
            annotated_objects=annotated_objects,
            normalized=normalized,
            bbox_mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "label_map": self.label_map,
            "annotated_objects": [o.to_dict() for o in self.annotated_objects]
            if self.annotated_objects is not None
            else None,
            "bbox_mode": self.bbox_mode.value,
            "normalized": self.normalized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectDetectionAnnotation:
        annotated_objects = data.get("annotated_objects")
        return cls(
            label_map=data["label_map"],
            annotated_objects=[AnnotatedObject.from_dict(o) for o in annotated_objects]
            if annotated_objects is not None
            else None,
            bbox_mode=BoundingBoxMode(data.get("bbox_mode", BoundingBoxMode.XYXY.value)),
            normalized=data.get("normalized", False),
        )


@dataclass(repr=False)
class LayoutAnalysisAnnotation(ObjectDetectionAnnotation):
    type = AnnotationType.layout_analysis.value


Annotation = (
    ClassificationAnnotation
    | EntityLabelingAnnotation
    | QuestionAnsweringAnnotation
    | ObjectDetectionAnnotation
    | LayoutAnalysisAnnotation
)


def annotation_from_dict(data: dict[str, Any]) -> Annotation:
    _map: dict[str, type[BaseDataModel]] = {
        AnnotationType.classification.value: ClassificationAnnotation,
        AnnotationType.entity_labeling.value: EntityLabelingAnnotation,
        AnnotationType.question_answering.value: QuestionAnsweringAnnotation,
        AnnotationType.object_detection.value: ObjectDetectionAnnotation,
        AnnotationType.layout_analysis.value: LayoutAnalysisAnnotation,
    }
    annotation_type = data.get("type")
    cls = _map.get(annotation_type) if annotation_type is not None else None
    if cls is None:
        raise ValueError(f"Unknown annotation type: {annotation_type!r}")
    return cast(
        "Annotation", cls.from_dict({k: v for k, v in data.items() if k != "type"})
    )
