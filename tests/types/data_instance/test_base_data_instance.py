from __future__ import annotations

import pytest

from atria_core.types._data_instance._base import (
    AnnotationNotFoundError,
    BaseDataInstance,
)
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
    instance = BaseDataInstance(sample_id="s1", annotations=[ann])
    assert instance.has_annotation_type(AnnotationType.classification) is True


def test_get_annotation_by_type_returns_match() -> None:
    ann = make_classification_annotation()
    instance = BaseDataInstance(sample_id="s1", annotations=[ann])
    assert instance.get_annotation_by_type(AnnotationType.classification) is ann


def test_get_annotation_by_type_raises_when_missing() -> None:
    instance = BaseDataInstance(sample_id="s1")
    with pytest.raises(AnnotationNotFoundError):
        instance.get_annotation_by_type(AnnotationType.classification)


def test_annotations_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        BaseDataInstance("s1", [])  # type: ignore[misc]


def test_equality() -> None:
    ann = make_classification_annotation()
    a = BaseDataInstance(sample_id="s1", annotations=[ann])
    b = BaseDataInstance(sample_id="s1", annotations=[ann])
    c = BaseDataInstance(sample_id="s2", annotations=[ann])
    assert a == b
    assert a != c
