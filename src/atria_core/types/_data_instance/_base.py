from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Self, overload

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._annotations import (
    ANNOTATION_TYPES,
    Annotation,
    AnnotationType,
    ClassificationAnnotation,
    EntityLabelingAnnotation,
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
    OCRAnnotation,
    QuestionAnsweringAnnotation,
    SentimentAnnotation,
    TranscriptionAnnotation,
)


@dataclass(frozen=True, repr=False)
class DataInstance(BaseDataModel):
    sample_id: str
    #: Keyed by annotation.type -- at most one annotation per type. Private:
    #: the only supported way to add/replace an entry is add_annotation(),
    #: and the only way to read one is has_annotation_type()/
    #: get_annotation_by_type() -- that's what keeps the key always in sync
    #: with the value's own type.
    _annotations: dict[str, Annotation] = field(default_factory=dict, kw_only=True)

    def __rich_repr__(self) -> Any:
        yield from super().__rich_repr__()
        if self._annotations:
            yield "annotations", self._annotations

    @property
    def key(self) -> str:
        # "#" is a URI fragment delimiter -- ResourceLoader.for_uri would
        # otherwise silently truncate a path built from this key at that
        # character (e.g. a multi-page document's "name#page_id" sample_id).
        return self.sample_id.replace(".", "_").replace("/", "_").replace("#", "_")

    # -------------------------------------
    # Annotation helpers
    # -------------------------------------
    def add_annotation(self, annotation: Annotation) -> Self:
        """Returns a new instance with `annotation` added, replacing any
        existing annotation of the same type."""
        return replace(
            self, _annotations={**self._annotations, annotation.type: annotation}
        )

    def has_annotation_type(self, annotation_type: AnnotationType) -> bool:
        return annotation_type.value in self._annotations

    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.classification]
    ) -> ClassificationAnnotation | None: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.entity_labeling]
    ) -> EntityLabelingAnnotation | None: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.question_answering]
    ) -> QuestionAnsweringAnnotation | None: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.object_detection]
    ) -> ObjectDetectionAnnotation | None: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.layout_analysis]
    ) -> LayoutAnalysisAnnotation | None: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.transcription]
    ) -> TranscriptionAnnotation | None: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.ocr]
    ) -> OCRAnnotation | None: ...
    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.sentiment]
    ) -> SentimentAnnotation | None: ...

    def get_annotation_by_type(
        self, annotation_type: AnnotationType
    ) -> Annotation | None:
        return self._annotations.get(annotation_type.value)

    # -------------------------------------
    # Shared (de)serialization helpers for subclasses
    # -------------------------------------
    def _annotations_to_dict(self) -> dict[str, dict[str, Any]] | None:
        if not self._annotations:
            return None
        return {t: a.to_dict() for t, a in self._annotations.items()}

    @staticmethod
    def _annotations_from_dict(
        data: dict[str, dict[str, Any]] | None,
    ) -> dict[str, Annotation]:
        if not data:
            return {}
        result: dict[str, Annotation] = {}
        for annotation_type, item in data.items():
            cls = ANNOTATION_TYPES.get(annotation_type)
            if cls is None:
                raise ValueError(f"Unknown annotation type: {annotation_type!r}")
            result[annotation_type] = cls.from_dict(item)
        return result
