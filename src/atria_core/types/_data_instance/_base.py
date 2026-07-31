from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, overload

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._annotations import (
    Annotation,
    AnnotationType,
    ClassificationAnnotation,
    EntityLabelingAnnotation,
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
    QuestionAnsweringAnnotation,
    annotation_from_dict,
)


class AnnotationNotFoundError(Exception):
    """Custom exception raised when a specific annotation type is not found in a data instance."""

    pass


@dataclass(repr=False)
class BaseDataInstance(BaseDataModel):
    sample_id: str
    annotations: list[Annotation] | None = field(default=None, kw_only=True)

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
    # Shared (de)serialization helpers for subclasses
    # -------------------------------------
    def _annotations_to_dict(self) -> list[dict[str, Any]] | None:
        if self.annotations is None:
            return None
        return [a.to_dict() for a in self.annotations]

    @staticmethod
    def _annotations_from_dict(
        data: list[dict[str, Any]] | None,
    ) -> list[Annotation] | None:
        if data is None:
            return None
        return [annotation_from_dict(a) for a in data]
