from __future__ import annotations

from atria_core.types._generic._annotated_object import AnnotatedObject
from tests.types.builders import make_annotated_object, make_bounding_box


def test_construct_defaults() -> None:
    ao = make_annotated_object()
    assert ao.label_value == 0
    assert ao.segmentation is None
    assert ao.iscrowd is False


def test_equality_is_identity_based() -> None:
    # AnnotatedObject holds numpy array fields, so it opts out of dataclass
    # equality (eq=False) rather than crash on `==`; only identity compares.
    bbox = make_bounding_box()
    a = AnnotatedObject(label_value=1, bbox=bbox)
    assert a == a
    b = AnnotatedObject(label_value=1, bbox=bbox)
    assert a != b


def test_to_dict_from_dict_roundtrip() -> None:
    ao = make_annotated_object(
        label_value=3, segmentation=[[1.0, 2.0], [3.0, 4.0]], iscrowd=True
    )
    data = ao.to_dict()
    restored = AnnotatedObject.from_dict(data)
    assert restored.to_dict() == data
