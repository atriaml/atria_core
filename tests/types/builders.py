from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image as PILImage

from atria_core.types._generic._annotated_object import AnnotatedObject
from atria_core.types._generic._annotations import (
    ClassificationAnnotation,
    EntityLabelingAnnotation,
    ObjectDetectionAnnotation,
    QuestionAnsweringAnnotation,
)
from atria_core.types._generic._doc_content import DocumentContent, TextElement
from atria_core.types._generic._image import Image
from atria_core.types._generic._qa_pair import QAPair


def make_bounding_box(**overrides: Any) -> np.ndarray:
    value = overrides.get("value", (0.1, 0.2, 0.5, 0.6))
    return np.asarray(value, dtype=np.float64)


def make_annotated_object(**overrides: Any) -> AnnotatedObject:
    kwargs: dict[str, Any] = {"label": 0, "bbox": make_bounding_box()}
    kwargs.update(overrides)
    return AnnotatedObject(**kwargs)


def make_text_element(**overrides: Any) -> TextElement:
    kwargs: dict[str, Any] = {
        "text": "hello",
        "bbox": make_bounding_box(),
        "conf": 0.95,
    }
    kwargs.update(overrides)
    return TextElement(**kwargs)


def make_document_content(**overrides: Any) -> DocumentContent:
    kwargs: dict[str, Any] = {"text_elements": [make_text_element()]}
    kwargs.update(overrides)
    return DocumentContent(**kwargs)


def make_qa_pair(**overrides: Any) -> QAPair:
    kwargs: dict[str, Any] = {
        "id": 0,
        "question_text": "what is this?",
        "answer_text": "this",
        "start": 0,
        "end": 4,
    }
    kwargs.update(overrides)
    return QAPair(**kwargs)


def make_classification_annotation(**overrides: Any) -> ClassificationAnnotation:
    kwargs: dict[str, Any] = {"label": 0, "label_map": ["cat", "dog"]}
    kwargs.update(overrides)
    return ClassificationAnnotation(**kwargs)


def make_entity_labeling_annotation(**overrides: Any) -> EntityLabelingAnnotation:
    kwargs: dict[str, Any] = {"word_labels": [0, 1, 0], "label_map": ["O", "B-ENT"]}
    kwargs.update(overrides)
    return EntityLabelingAnnotation(**kwargs)


def make_question_answering_annotation(**overrides: Any) -> QuestionAnsweringAnnotation:
    kwargs: dict[str, Any] = {"qa_pairs": [make_qa_pair()]}
    kwargs.update(overrides)
    return QuestionAnsweringAnnotation(**kwargs)


def make_object_detection_annotation(**overrides: Any) -> ObjectDetectionAnnotation:
    kwargs: dict[str, Any] = {
        "label_map": ["cat", "dog"],
        "annotated_objects": [make_annotated_object()],
    }
    kwargs.update(overrides)
    return ObjectDetectionAnnotation(**kwargs)


def make_pil_image(size: tuple[int, int] = (16, 12), color: str = "white") -> PILImage.Image:
    return PILImage.new("RGB", size, color=color)


def make_image(**overrides: Any) -> Image:
    source = overrides.pop("source", None) or make_pil_image()
    return Image(source)
