from __future__ import annotations

import pytest

from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._generic._annotations import AnnotationType
from tests.types.builders import make_classification_annotation


def test_key_replaces_dots_and_slashes() -> None:
    instance = BaseDataInstance(sample_id="a/b.c")
    assert instance.key == "a_b_c"


def test_has_annotation_type_false_when_none() -> None:
    instance = BaseDataInstance(sample_id="s1")
    assert instance.has_annotation_type(AnnotationType.classification) is False


def test_has_annotation_type_true_when_present() -> None:
    ann = make_classification_annotation()
    instance = BaseDataInstance(sample_id="s1").add_annotation(ann)
    assert instance.has_annotation_type(AnnotationType.classification) is True


def test_get_annotation_by_type_returns_match() -> None:
    ann = make_classification_annotation()
    instance = BaseDataInstance(sample_id="s1").add_annotation(ann)
    assert instance.get_annotation_by_type(AnnotationType.classification) is ann


def test_get_annotation_by_type_returns_none_when_missing() -> None:
    instance = BaseDataInstance(sample_id="s1")
    assert instance.get_annotation_by_type(AnnotationType.classification) is None


def test_add_annotation_returns_new_instance_and_replaces_same_type() -> None:
    first = make_classification_annotation(label_value=0)
    second = make_classification_annotation(label_value=1)
    instance = BaseDataInstance(sample_id="s1")

    with_first = instance.add_annotation(first)
    assert instance.get_annotation_by_type(AnnotationType.classification) is None
    assert with_first.get_annotation_by_type(AnnotationType.classification) is first

    with_second = with_first.add_annotation(second)
    assert with_second.get_annotation_by_type(AnnotationType.classification) is second
    # add_annotation doesn't mutate the instance it was called on.
    assert with_first.get_annotation_by_type(AnnotationType.classification) is first


def test_annotations_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        BaseDataInstance("s1", {})  # type: ignore[misc]


def test_equality() -> None:
    ann = make_classification_annotation()
    a = BaseDataInstance(sample_id="s1").add_annotation(ann)
    b = BaseDataInstance(sample_id="s1").add_annotation(ann)
    c = BaseDataInstance(sample_id="s2").add_annotation(ann)
    assert a == b
    assert a != c


def test_annotations_to_dict_from_dict_roundtrip() -> None:
    ann = make_classification_annotation()
    instance = BaseDataInstance(sample_id="s1").add_annotation(ann)
    data = instance._annotations_to_dict()
    assert data == {"classification": ann.to_dict()}
    restored = BaseDataInstance._annotations_from_dict(data)
    assert restored["classification"].to_dict() == ann.to_dict()


def test_annotations_to_dict_is_none_when_empty() -> None:
    instance = BaseDataInstance(sample_id="s1")
    assert instance._annotations_to_dict() is None


def test_annotations_from_dict_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown annotation type"):
        BaseDataInstance._annotations_from_dict({"not_a_real_type": {}})
