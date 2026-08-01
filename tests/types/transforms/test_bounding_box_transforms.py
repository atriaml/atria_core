from __future__ import annotations

import numpy as np

from atria_core.types._generic._annotated_object import AnnotatedObject
from atria_core.types._generic._annotations import (
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
)
from atria_core.types._generic._bounding_box import BoundingBoxMode
from atria_core.types._transforms._bounding_box import BoundingBoxTransformer

# BoundingBoxTransformer applies to any BoxBatchOwner -- currently
# ObjectDetectionAnnotation (and its subclass LayoutAnalysisAnnotation).
# DocumentContent/ElementArray bboxes are always normalized by design (see
# _generic/_elements.py) and don't implement this protocol.


def _annotation(bbox=(10, 10, 50, 60)) -> ObjectDetectionAnnotation:
    return ObjectDetectionAnnotation.from_objects(
        [AnnotatedObject(label=0, bbox=np.asarray(bbox, dtype=np.float64))], label_map=["a"]
    )


def test_normalize() -> None:
    ann = _annotation()
    result = BoundingBoxTransformer.normalize(ann, 100, 100)
    assert result.normalized is True
    assert np.array_equal(result.bboxes[0], [0.1, 0.1, 0.5, 0.6])


def test_normalize_is_noop_when_already_normalized() -> None:
    ann = _annotation()
    once = BoundingBoxTransformer.normalize(ann, 100, 100)
    twice = BoundingBoxTransformer.normalize(once, 100, 100)
    assert twice is once


def test_unnormalize_reverses_normalize() -> None:
    ann = _annotation()
    normalized = BoundingBoxTransformer.normalize(ann, 100, 100)
    restored = BoundingBoxTransformer.unnormalize(normalized, 100, 100)
    assert restored.normalized is False
    assert np.array_equal(restored.bboxes[0], [10.0, 10.0, 50.0, 60.0])


def test_unnormalize_is_noop_when_not_normalized() -> None:
    ann = _annotation()
    result = BoundingBoxTransformer.unnormalize(ann, 100, 100)
    assert result is ann


def test_switch_mode() -> None:
    ann = _annotation(bbox=(10, 20, 50, 60))
    switched = BoundingBoxTransformer.switch_mode(ann)
    assert switched.bbox_mode == BoundingBoxMode.XYWH
    assert np.array_equal(switched.bboxes[0], [10, 20, 40, 40])


def test_switch_mode_roundtrip() -> None:
    ann = _annotation(bbox=(10, 20, 50, 60))
    roundtripped = BoundingBoxTransformer.switch_mode(BoundingBoxTransformer.switch_mode(ann))
    assert roundtripped.bbox_mode == ann.bbox_mode
    assert np.array_equal(roundtripped.bboxes[0], ann.bboxes[0])


def test_composed_normalize_and_switch_mode() -> None:
    x = _annotation(bbox=(10, 10, 50, 60))
    x = BoundingBoxTransformer.normalize(x, 100, 100)
    x = BoundingBoxTransformer.switch_mode(x)
    assert x.normalized is True
    assert x.bbox_mode == BoundingBoxMode.XYWH


def test_normalize_with_no_objects_is_noop() -> None:
    ann = ObjectDetectionAnnotation(label_map=["a"])
    result = BoundingBoxTransformer.normalize(ann, 100, 100)
    assert result is ann


def test_normalize_preserves_layout_analysis_annotation_subclass() -> None:
    ann = LayoutAnalysisAnnotation.from_objects(
        [AnnotatedObject(label=0, bbox=np.array([10.0, 10.0, 50.0, 60.0]))], label_map=["a"]
    )
    result = BoundingBoxTransformer.normalize(ann, 100, 100)
    assert isinstance(result, LayoutAnalysisAnnotation)
