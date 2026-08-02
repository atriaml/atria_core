from __future__ import annotations

import numpy as np

from atria_core.transforms import functional as F
from atria_core.types import (
    AnnotatedObject,
    BoundingBoxMode,
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
)

# F.bbox applies to any BoxBatchOwner -- currently ObjectDetectionAnnotation
# (and its subclass LayoutAnalysisAnnotation). DocumentContent/ElementArray
# bboxes are always normalized by design (see atria_core.types._generic._elements)
# and don't implement this protocol.


def _annotation(bbox=(10, 10, 50, 60)) -> ObjectDetectionAnnotation:
    return ObjectDetectionAnnotation.from_objects(
        [AnnotatedObject(label=0, bbox=np.asarray(bbox, dtype=np.float64))],
        label_map=["a"],
    )


def test_normalize() -> None:
    ann = _annotation()
    result = F.bbox.normalize(ann, 100, 100)
    assert result.normalized is True
    assert np.array_equal(result.bboxes[0], [0.1, 0.1, 0.5, 0.6])


def test_normalize_is_noop_when_already_normalized() -> None:
    ann = _annotation()
    once = F.bbox.normalize(ann, 100, 100)
    twice = F.bbox.normalize(once, 100, 100)
    assert twice is once


def test_unnormalize_reverses_normalize() -> None:
    ann = _annotation()
    normalized = F.bbox.normalize(ann, 100, 100)
    restored = F.bbox.unnormalize(normalized, 100, 100)
    assert restored.normalized is False
    assert np.array_equal(restored.bboxes[0], [10.0, 10.0, 50.0, 60.0])


def test_unnormalize_is_noop_when_not_normalized() -> None:
    ann = _annotation()
    result = F.bbox.unnormalize(ann, 100, 100)
    assert result is ann


def test_switch_mode() -> None:
    ann = _annotation(bbox=(10, 20, 50, 60))
    switched = F.bbox.switch_mode(ann)
    assert switched.bbox_mode == BoundingBoxMode.XYWH
    assert np.array_equal(switched.bboxes[0], [10, 20, 40, 40])


def test_switch_mode_roundtrip() -> None:
    ann = _annotation(bbox=(10, 20, 50, 60))
    roundtripped = F.bbox.switch_mode(F.bbox.switch_mode(ann))
    assert roundtripped.bbox_mode == ann.bbox_mode
    assert np.array_equal(roundtripped.bboxes[0], ann.bboxes[0])


def test_composed_normalize_and_switch_mode() -> None:
    x = _annotation(bbox=(10, 10, 50, 60))
    x = F.bbox.normalize(x, 100, 100)
    x = F.bbox.switch_mode(x)
    assert x.normalized is True
    assert x.bbox_mode == BoundingBoxMode.XYWH


def test_normalize_with_no_objects_is_noop() -> None:
    ann = ObjectDetectionAnnotation(label_map=["a"])
    result = F.bbox.normalize(ann, 100, 100)
    assert result is ann


def test_normalize_preserves_layout_analysis_annotation_subclass() -> None:
    ann = LayoutAnalysisAnnotation.from_objects(
        [AnnotatedObject(label=0, bbox=np.array([10.0, 10.0, 50.0, 60.0]))],
        label_map=["a"],
    )
    result = F.bbox.normalize(ann, 100, 100)
    assert isinstance(result, LayoutAnalysisAnnotation)
