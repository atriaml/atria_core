from __future__ import annotations

import enum
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._annotated_object import AnnotatedObject
from atria_core.types._generic._bounding_box import BoundingBoxMode
from atria_core.types._generic._elements import ElementArray, OCRLevel
from atria_core.types._generic._qa_pair import QAPair


class AnnotationType(str, enum.Enum):
    classification = "classification"
    entity_labeling = "entity_labeling"
    question_answering = "question_answering"
    object_detection = "object_detection"
    layout_analysis = "layout_analysis"
    ocr = "ocr"
    transcription = "transcription"


@dataclass(frozen=True, repr=False)
class ClassificationAnnotation(BaseDataModel):
    type = AnnotationType.classification.value

    label_value: int
    label_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "label_value": self.label_value,
            "label_name": self.label_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationAnnotation:
        return cls(label_value=data["label_value"], label_name=data["label_name"])


@dataclass(frozen=True, repr=False)
class EntityLabelingAnnotation(BaseDataModel):
    type = AnnotationType.entity_labeling.value

    word_label_values: list[int]
    word_label_names: list[str]

    def serialize_word_labels(self) -> str:
        return json.dumps(self.word_label_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "word_label_values": self.word_label_values,
            "word_label_names": self.word_label_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityLabelingAnnotation:
        return cls(
            word_label_values=list(data["word_label_values"]),
            word_label_names=data["word_label_names"],
        )


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

    label_values: np.ndarray | None = None
    label_names: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    segmentations: np.ndarray | None = None  # (N, P_max, 2), NaN-padded
    segmentation_lengths: np.ndarray | None = None  # (N,) real point count per object
    iscrowd: np.ndarray | None = None
    bbox_mode: BoundingBoxMode = BoundingBoxMode.XYXY
    normalized: bool = False

    @classmethod
    def from_objects(cls, objects: list[AnnotatedObject]) -> ObjectDetectionAnnotation:
        """The human-readable construction path: build one AnnotatedObject
        per detection, then stack them into this annotation's arrays."""
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
            label_values=np.array([o.label_value for o in objects]),
            label_names=np.array([o.label_name for o in objects]),
            bboxes=np.stack([o.bbox for o in objects]),
            segmentations=segmentations,
            segmentation_lengths=segmentation_lengths,
            iscrowd=np.array([o.iscrowd for o in objects]),
        )

    def to_objects(self) -> list[AnnotatedObject]:
        """The human-readable view: one AnnotatedObject per row."""
        if self.label_values is None or self.bboxes is None:
            return []
        n = len(self.label_values)
        iscrowd = self.iscrowd if self.iscrowd is not None else np.zeros(n, dtype=bool)

        def _segmentation(i: int) -> np.ndarray | None:
            if self.segmentations is None or self.segmentation_lengths is None:
                return None
            length = int(self.segmentation_lengths[i])
            return self.segmentations[i, :length] if length > 0 else None

        return [
            AnnotatedObject(
                label_value=int(self.label_values[i]),
                label_name=str(self.label_names[i]),
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
            "label_names": self.label_names.tolist()
            if self.label_names is not None
            else None,
            "label_values": self.label_values.tolist()
            if self.label_values is not None
            else None,
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
            label_names=_array("label_names", object),
            label_values=_array("label_values", np.int64),
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


@dataclass(frozen=True, repr=False)
class TranscriptionAnnotation(BaseDataModel):
    type = AnnotationType.transcription.value

    text: str | None = None
    level: OCRLevel | None = None

    def __post_init__(self):
        assert isinstance(self.text, str | None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptionAnnotation:
        return cls(
            text=data.get("text"),
        )


@dataclass(frozen=True, repr=False, eq=False)
class OCRAnnotation(ElementArray):
    """Ground-truth OCR/text-layer hierarchy for a document instance --
    same structure-of-arrays shape as ElementArray (page/block/line/word,
    with parent links, normalized bboxes, and optional polygons), just
    tagged as an `ocr` annotation so it round-trips through
    BaseDataInstance's annotation dict. A single instance can carry the
    whole hierarchy at once (sliceable via `.at(level)`), which is what
    lets one document hold word-level *and* line-level ground truth
    together."""

    type = AnnotationType.ocr.value


Annotation = (
    ClassificationAnnotation
    | EntityLabelingAnnotation
    | QuestionAnsweringAnnotation
    | ObjectDetectionAnnotation
    | LayoutAnalysisAnnotation
    | TranscriptionAnnotation
    | OCRAnnotation
)

#: type string -> class, for BaseDataInstance's annotation dict (de)serialization.
ANNOTATION_TYPES: dict[str, type[Annotation]] = {
    AnnotationType.classification.value: ClassificationAnnotation,
    AnnotationType.entity_labeling.value: EntityLabelingAnnotation,
    AnnotationType.question_answering.value: QuestionAnsweringAnnotation,
    AnnotationType.object_detection.value: ObjectDetectionAnnotation,
    AnnotationType.layout_analysis.value: LayoutAnalysisAnnotation,
    AnnotationType.transcription.value: TranscriptionAnnotation,
    AnnotationType.ocr.value: OCRAnnotation,
}
