from __future__ import annotations

import numpy as np
import pytest

from atria_core.types._generic._annotated_object import AnnotatedObject
from atria_core.types._generic._annotations import (
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
)
from atria_core.types._generic._bounding_box import BoundingBoxMode
from atria_core.types._generic._doc_content import DocumentContent, TextElement
from atria_core.types._transforms._bounding_box import BoundingBoxTransformer


def _content(bbox=(10, 10, 50, 60), segment_bbox=(0, 0, 100, 100)) -> DocumentContent:
    return DocumentContent(
        text_elements=[TextElement(text="a", bbox=bbox, segment_bbox=segment_bbox)]
    )


def _annotation(bbox=(10, 10, 50, 60)) -> ObjectDetectionAnnotation:
    return ObjectDetectionAnnotation(
        label_map=["a"], annotated_objects=[AnnotatedObject(label=0, bbox=bbox)]
    )


CONTAINER_FACTORIES = [_content, _annotation]


def _boxes(container) -> np.ndarray:
    if isinstance(container, DocumentContent):
        return container.text_elements[0].bbox
    return container.annotated_objects[0].bbox


@pytest.mark.parametrize("make_container", CONTAINER_FACTORIES)
def test_normalize(make_container) -> None:
    container = make_container()
    result = BoundingBoxTransformer.normalize(container, 100, 100)
    assert result.normalized is True
    assert np.array_equal(_boxes(result), [0.1, 0.1, 0.5, 0.6])


@pytest.mark.parametrize("make_container", CONTAINER_FACTORIES)
def test_normalize_is_noop_when_already_normalized(make_container) -> None:
    container = make_container()
    once = BoundingBoxTransformer.normalize(container, 100, 100)
    twice = BoundingBoxTransformer.normalize(once, 100, 100)
    assert twice is once


@pytest.mark.parametrize("make_container", CONTAINER_FACTORIES)
def test_unnormalize_reverses_normalize(make_container) -> None:
    container = make_container()
    normalized = BoundingBoxTransformer.normalize(container, 100, 100)
    restored = BoundingBoxTransformer.unnormalize(normalized, 100, 100)
    assert restored.normalized is False
    assert np.array_equal(_boxes(restored), [10.0, 10.0, 50.0, 60.0])


@pytest.mark.parametrize("make_container", CONTAINER_FACTORIES)
def test_unnormalize_is_noop_when_not_normalized(make_container) -> None:
    container = make_container()
    result = BoundingBoxTransformer.unnormalize(container, 100, 100)
    assert result is container


@pytest.mark.parametrize("make_container", CONTAINER_FACTORIES)
def test_switch_mode(make_container) -> None:
    container = make_container(bbox=(10, 20, 50, 60))
    switched = BoundingBoxTransformer.switch_mode(container)
    assert switched.bbox_mode == BoundingBoxMode.XYWH
    assert np.array_equal(_boxes(switched), [10, 20, 40, 40])


@pytest.mark.parametrize("make_container", CONTAINER_FACTORIES)
def test_switch_mode_roundtrip(make_container) -> None:
    container = make_container(bbox=(10, 20, 50, 60))
    roundtripped = BoundingBoxTransformer.switch_mode(
        BoundingBoxTransformer.switch_mode(container)
    )
    assert roundtripped.bbox_mode == container.bbox_mode
    assert np.array_equal(_boxes(roundtripped), _boxes(container))


def test_composed_normalize_and_switch_mode() -> None:
    x = _content(bbox=(10, 10, 50, 60))
    x = BoundingBoxTransformer.normalize(x, 100, 100)
    x = BoundingBoxTransformer.switch_mode(x)
    assert x.normalized is True
    assert x.bbox_mode == BoundingBoxMode.XYWH


def test_normalize_document_content_none_bbox_untouched() -> None:
    content = DocumentContent(text_elements=[TextElement(text="a")])
    result = BoundingBoxTransformer.normalize(content, 100, 100)
    assert result.text_elements[0].bbox is None


def test_normalize_preserves_layout_analysis_annotation_subclass() -> None:
    ann = LayoutAnalysisAnnotation(
        label_map=["a"], annotated_objects=[AnnotatedObject(label=0, bbox=(10, 10, 50, 60))]
    )
    result = BoundingBoxTransformer.normalize(ann, 100, 100)
    assert isinstance(result, LayoutAnalysisAnnotation)
