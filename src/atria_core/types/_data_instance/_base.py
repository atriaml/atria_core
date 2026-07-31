from __future__ import annotations

from typing import Literal, overload

from atria_core.types._generic._annotations import (
    Annotation,
    AnnotationType,
    ClassificationAnnotation,
    EntityLabelingAnnotation,
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
    QuestionAnsweringAnnotation,
)


class AnnotationNotFoundError(Exception):
    """Custom exception raised when a specific annotation type is not found in a data instance."""

    pass


class BaseDataInstance:
    def __init__(
        self,
        sample_id: str,
        annotations: list[Annotation] | None = None,
    ) -> None:
        self.sample_id = sample_id
        self.annotations = annotations

    @property
    def key(self) -> str:
        return self.sample_id.replace(".", "_").replace("/", "_")

    # -------------------------------------
    # Annotation helpers
    # -------------------------------------
    def has_annotation_type(self, annotation_type: AnnotationType) -> bool:
        if self.annotations is not None:
            for annotation in self.annotations:
                if annotation.type == annotation_type.value:
                    return True
        return False

    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.classification]
    ) -> ClassificationAnnotation: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.entity_labeling]
    ) -> EntityLabelingAnnotation: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.question_answering]
    ) -> QuestionAnsweringAnnotation: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.object_detection]
    ) -> ObjectDetectionAnnotation: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.layout_analysis]
    ) -> LayoutAnalysisAnnotation: ...

    def get_annotation_by_type(self, annotation_type: AnnotationType) -> Annotation:
        if self.annotations is not None:
            for annotation in self.annotations:
                if annotation.type == annotation_type.value:
                    return annotation
        raise AnnotationNotFoundError(
            f"No annotation of type {annotation_type} found in the data instance."
        )

    # -------------------------------------
    # Dunder helpers
    # -------------------------------------
    def __repr__(self) -> str:
        n = len(self.annotations) if self.annotations else 0
        return f"BaseDataInstance(sample_id={self.sample_id!r}, index={self.index}, annotations={n})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseDataInstance):
            return NotImplemented
        return (
            self.sample_id == other.sample_id
            and self.index == other.index
            and self.annotations == other.annotations
        )
