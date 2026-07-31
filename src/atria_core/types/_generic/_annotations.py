from __future__ import annotations

import enum
import json

from atria_core.types._generic._annotated_object import AnnotatedObject
from atria_core.types._generic._qa_pair import QAPair


class AnnotationType(str, enum.Enum):
    classification = "classification"
    entity_labeling = "entity_labeling"
    question_answering = "question_answering"
    object_detection = "object_detection"
    layout_analysis = "layout_analysis"


# index = label value, value = label name
LabelMap = list[str]  # ["cat", "dog", ...]


class ClassificationAnnotation:
    type = AnnotationType.classification.value

    def __init__(self, label: int, label_map: LabelMap) -> None:
        self.label = label
        self.label_map = label_map

    @property
    def label_name(self) -> str:
        return self.label_map[self.label]

    def __repr__(self) -> str:
        return f"ClassificationAnnotation(label={self.label}, label_name={self.label_name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassificationAnnotation):
            return NotImplemented
        return self.label == other.label and self.label_map == other.label_map


class EntityLabelingAnnotation:
    type = AnnotationType.entity_labeling.value

    def __init__(self, word_labels: list[int] | str, label_map: LabelMap) -> None:
        self.word_labels = self._parse_word_labels(word_labels)
        self.label_map = label_map

    @staticmethod
    def _parse_word_labels(value: list[int] | str) -> list[int]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return list(value)

    @property
    def label_names(self) -> list[str]:
        return [self.label_map[l] for l in self.word_labels]

    def serialize_word_labels(self) -> str:
        return json.dumps(self.word_labels)

    def __repr__(self) -> str:
        return f"EntityLabelingAnnotation(word_labels={self.word_labels!r}, label_names={self.label_names!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntityLabelingAnnotation):
            return NotImplemented
        return (
            self.word_labels == other.word_labels and self.label_map == other.label_map
        )


class QuestionAnsweringAnnotation:
    type = AnnotationType.question_answering.value

    def __init__(self, qa_pairs: list[QAPair] | str) -> None:
        self.qa_pairs = self._parse_qa_pairs(qa_pairs)

    @staticmethod
    def _parse_qa_pairs(value: list[QAPair] | str) -> list[QAPair]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return [QAPair(**item) if isinstance(item, dict) else item for item in value]

    def serialize_qa_pairs(self) -> str:
        return json.dumps(
            [
                {
                    "id": q.id,
                    "question_text": q.question_text,
                    "answer_spans": q.serialize_answer_spans(),
                }
                for q in self.qa_pairs
            ]
        )

    def __repr__(self) -> str:
        return f"QuestionAnsweringAnnotation(qa_pairs={self.qa_pairs!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuestionAnsweringAnnotation):
            return NotImplemented
        return self.qa_pairs == other.qa_pairs


class ObjectDetectionAnnotation:
    type = AnnotationType.object_detection.value

    def __init__(
        self,
        label_map: LabelMap,
        annotated_objects: list[AnnotatedObject] | str | None = None,
    ) -> None:
        self.label_map = label_map
        self.annotated_objects = self._parse_annotated_objects(annotated_objects)

    @staticmethod
    def _parse_annotated_objects(
        value: list[AnnotatedObject] | str | None,
    ) -> list[AnnotatedObject] | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return [
            AnnotatedObject(**item) if isinstance(item, dict) else item
            for item in value
        ]

    def serialize_annotated_objects(self) -> str | None:
        if self.annotated_objects is None:
            return None
        return json.dumps(
            [
                {
                    "label": o.label,
                    "bbox": {
                        "value": o.bbox.value,
                        "mode": o.bbox.mode.value,
                        "normalized": o.bbox.normalized,
                    },
                    "segmentation": o.segmentation,
                    "iscrowd": o.iscrowd,
                }
                for o in self.annotated_objects
            ]
        )

    def __repr__(self) -> str:
        return (
            f"ObjectDetectionAnnotation(annotated_objects={self.annotated_objects!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectDetectionAnnotation):
            return NotImplemented
        return (
            self.annotated_objects == other.annotated_objects
            and self.label_map == other.label_map
        )


class LayoutAnalysisAnnotation(ObjectDetectionAnnotation):
    type = AnnotationType.layout_analysis.value

    def __repr__(self) -> str:
        return f"LayoutAnalysisAnnotation(annotated_objects={self.annotated_objects!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayoutAnalysisAnnotation):
            return NotImplemented
        return (
            self.annotated_objects == other.annotated_objects
            and self.label_map == other.label_map
        )


Annotation = (
    ClassificationAnnotation
    | EntityLabelingAnnotation
    | QuestionAnsweringAnnotation
    | ObjectDetectionAnnotation
    | LayoutAnalysisAnnotation
)


def annotation_from_dict(data: dict) -> Annotation:
    _map = {
        AnnotationType.classification.value: ClassificationAnnotation,
        AnnotationType.entity_labeling.value: EntityLabelingAnnotation,
        AnnotationType.question_answering.value: QuestionAnsweringAnnotation,
        AnnotationType.object_detection.value: ObjectDetectionAnnotation,
        AnnotationType.layout_analysis.value: LayoutAnalysisAnnotation,
    }
    annotation_type = data.get("type")
    cls = _map.get(annotation_type)
    if cls is None:
        raise ValueError(f"Unknown annotation type: {annotation_type!r}")
    return cls(**{k: v for k, v in data.items() if k != "type"})
