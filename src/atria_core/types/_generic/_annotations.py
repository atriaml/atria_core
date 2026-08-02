from __future__ import annotations

import enum
import json
from dataclasses import dataclass, replace
from typing import Any

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


@dataclass(frozen=True, repr=False)
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


@dataclass(frozen=True, repr=False)
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


@dataclass(frozen=True, repr=False)
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


@dataclass(frozen=True, repr=False, eq=False)
class ObjectDetectionAnnotation(BaseDataModel):
    """Detected objects, stored as structure-of-arrays: one entry per
    object, but columnar (parallel arrays) rather than a list of per-object
    objects. Polygon point counts are ragged, so `segmentations` is one
    padded (N, P_max, 2) array (NaN-padded) plus `segmentation_lengths` for
    each object's real point count -- avoids per-object boxed arrays while
    staying genuinely vectorizable. A given array field is present for every
    object or None for the whole annotation -- no per-object holes."""

    type = AnnotationType.object_detection.value

    label_map: list[str]
    labels: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    segmentations: np.ndarray | None = None  # (N, P_max, 2), NaN-padded
    segmentation_lengths: np.ndarray | None = None  # (N,) real point count per object
    iscrowd: np.ndarray | None = None
    bbox_mode: BoundingBoxMode = BoundingBoxMode.XYXY
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.labels is not None and self.labels.size > 0:
            if bool(np.any((self.labels < 0) | (self.labels >= len(self.label_map)))):
                bad = int(
                    self.labels[
                        (self.labels < 0) | (self.labels >= len(self.label_map))
                    ][0]
                )
                raise ValueError(
                    f"Invalid object label index {bad}. "
                    f"Label map contains only {len(self.label_map)} labels."
                )

    @classmethod
    def from_objects(
        cls, objects: list[AnnotatedObject], label_map: list[str]
    ) -> ObjectDetectionAnnotation:
        """The human-readable construction path: build one AnnotatedObject
        per detection, then stack them into this annotation's arrays."""
        if not objects:
            return cls(label_map=label_map)

        segmentations = None
        segmentation_lengths = None
        polygons = [o.segmentation for o in objects]
        if any(p is not None for p in polygons):
            lengths = np.array([0 if p is None else len(p) for p in polygons])
            p_max = int(lengths.max())
            padded = np.full((len(objects), p_max, 2), np.nan)
            for i, polygon in enumerate(polygons):
                if polygon is not None:
                    padded[i, : len(polygon)] = polygon
            segmentations = padded
            segmentation_lengths = lengths

        return cls(
            label_map=label_map,
            labels=np.array([o.label for o in objects]),
            bboxes=np.stack([o.bbox for o in objects]),
            segmentations=segmentations,
            segmentation_lengths=segmentation_lengths,
            iscrowd=np.array([o.iscrowd for o in objects]),
        )

    def to_objects(self) -> list[AnnotatedObject]:
        """The human-readable view: one AnnotatedObject per row."""
        if self.labels is None or self.bboxes is None:
            return []
        n = len(self.labels)
        iscrowd = self.iscrowd if self.iscrowd is not None else np.zeros(n, dtype=bool)

        def _segmentation(i: int) -> np.ndarray | None:
            if self.segmentations is None or self.segmentation_lengths is None:
                return None
            length = int(self.segmentation_lengths[i])
            return self.segmentations[i, :length] if length > 0 else None

        return [
            AnnotatedObject(
                label=int(self.labels[i]),
                bbox=self.bboxes[i],
                segmentation=_segmentation(i),
                iscrowd=bool(iscrowd[i]),
            )
            for i in range(n)
        ]

    # -------------------------------------
    # Generic batch-transform protocol (see _transforms/_bounding_box.py)
    # -------------------------------------
    def box_batches(self) -> dict[str, np.ndarray | None]:
        return {"bboxes": self.bboxes}

    def with_box_batches(
        self, batches: dict[str, np.ndarray], *, normalized: bool, mode: BoundingBoxMode
    ) -> ObjectDetectionAnnotation:
        return replace(
            self,
            bboxes=batches.get("bboxes", self.bboxes),
            normalized=normalized,
            bbox_mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "label_map": self.label_map,
            "labels": self.labels.tolist() if self.labels is not None else None,
            "bboxes": self.bboxes.tolist() if self.bboxes is not None else None,
            "segmentations": self.segmentations.tolist()
            if self.segmentations is not None
            else None,
            "segmentation_lengths": self.segmentation_lengths.tolist()
            if self.segmentation_lengths is not None
            else None,
            "iscrowd": self.iscrowd.tolist() if self.iscrowd is not None else None,
            "bbox_mode": self.bbox_mode.value,
            "normalized": self.normalized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectDetectionAnnotation:
        def _array(key: str, dtype: type) -> np.ndarray | None:
            value = data.get(key)
            return np.asarray(value, dtype=dtype) if value is not None else None

        return cls(
            label_map=data["label_map"],
            labels=_array("labels", np.int64),
            bboxes=_array("bboxes", np.float64),
            segmentations=_array("segmentations", np.float64),
            segmentation_lengths=_array("segmentation_lengths", np.int64),
            iscrowd=_array("iscrowd", np.bool_),
            bbox_mode=BoundingBoxMode(
                data.get("bbox_mode", BoundingBoxMode.XYXY.value)
            ),
            normalized=data.get("normalized", False),
        )


@dataclass(frozen=True, repr=False, eq=False)
class LayoutAnalysisAnnotation(ObjectDetectionAnnotation):
    type = AnnotationType.layout_analysis.value


Annotation = (
    ClassificationAnnotation
    | EntityLabelingAnnotation
    | QuestionAnsweringAnnotation
    | ObjectDetectionAnnotation
    | LayoutAnalysisAnnotation
)

#: type string -> class, for BaseDataInstance's annotation dict (de)serialization.
ANNOTATION_TYPES: dict[str, type[Annotation]] = {
    AnnotationType.classification.value: ClassificationAnnotation,
    AnnotationType.entity_labeling.value: EntityLabelingAnnotation,
    AnnotationType.question_answering.value: QuestionAnsweringAnnotation,
    AnnotationType.object_detection.value: ObjectDetectionAnnotation,
    AnnotationType.layout_analysis.value: LayoutAnalysisAnnotation,
}
