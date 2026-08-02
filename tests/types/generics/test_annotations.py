from __future__ import annotations

import numpy as np
import pytest

from atria_core.types._generic._annotations import (
    AnnotationType,
    ClassificationAnnotation,
    EntityLabelingAnnotation,
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
    QuestionAnsweringAnnotation,
)
from tests.types.builders import (
    make_annotated_object,
    make_classification_annotation,
    make_entity_labeling_annotation,
    make_object_detection_annotation,
    make_qa_pair,
    make_question_answering_annotation,
)


def test_classification_annotation_label_name() -> None:
    ann = make_classification_annotation(label=1, label_map=["cat", "dog"])
    assert ann.label_name == "dog"


def test_classification_annotation_rejects_out_of_range_label() -> None:
    with pytest.raises(ValueError, match="Invalid label index"):
        ClassificationAnnotation(label=5, label_map=["cat", "dog"])


def test_entity_labeling_annotation_label_names() -> None:
    ann = make_entity_labeling_annotation(word_labels=[0, 1], label_map=["O", "B-ENT"])
    assert ann.label_names == ["O", "B-ENT"]


def test_entity_labeling_annotation_rejects_out_of_range_label() -> None:
    with pytest.raises(ValueError, match="Invalid word label index"):
        EntityLabelingAnnotation(word_labels=[0, 9], label_map=["O", "B-ENT"])


def test_entity_labeling_annotation_serialize_word_labels() -> None:
    import json

    ann = make_entity_labeling_annotation(word_labels=[0, 1, 0])
    assert json.loads(ann.serialize_word_labels()) == [0, 1, 0]


def test_question_answering_annotation_roundtrip() -> None:
    ann = make_question_answering_annotation(
        qa_pairs=[make_qa_pair(), make_qa_pair(id=2)]
    )
    data = ann.to_dict()
    restored = QuestionAnsweringAnnotation.from_dict(data)
    assert restored == ann


def test_object_detection_annotation_from_objects_and_to_objects() -> None:
    objects = [make_annotated_object(label=0), make_annotated_object(label=1)]
    ann = ObjectDetectionAnnotation.from_objects(objects, label_map=["cat", "dog"])

    assert list(ann.labels) == [0, 1]
    assert ann.bboxes.shape == (2, 4)

    back = ann.to_objects()
    assert [o.label for o in back] == [0, 1]


def test_object_detection_annotation_segmentation_padding_mixed_objects() -> None:
    # some objects have a segmentation (varying point counts), some don't --
    # segmentations must pad to the max point count and track real lengths.
    objects = [
        make_annotated_object(segmentation=[[0.1, 0.1], [0.2, 0.2], [0.3, 0.1]]),
        make_annotated_object(label=1),  # no segmentation
        make_annotated_object(segmentation=[[0.0, 0.0], [0.05, 0.05]]),
    ]
    ann = ObjectDetectionAnnotation.from_objects(objects, label_map=["a", "b"])

    assert ann.segmentations.shape == (3, 3, 2)
    assert list(ann.segmentation_lengths) == [3, 0, 2]

    back = ann.to_objects()
    assert back[0].segmentation.shape == (3, 2)
    assert back[1].segmentation is None
    assert back[2].segmentation.shape == (2, 2)

    data = ann.to_dict()
    restored = ObjectDetectionAnnotation.from_dict(data)
    # NaN padding means to_dict() output isn't self-equal via `==`; compare
    # everything except the NaN-bearing field directly, and that one with
    # equal_nan.
    restored_data = restored.to_dict()
    assert {k: v for k, v in restored_data.items() if k != "segmentations"} == {
        k: v for k, v in data.items() if k != "segmentations"
    }
    assert np.array_equal(restored.segmentations, ann.segmentations, equal_nan=True)
    restored_objects = restored.to_objects()
    assert restored_objects[1].segmentation is None
    assert restored_objects[2].segmentation.shape == (2, 2)


def test_object_detection_annotation_roundtrip() -> None:
    ann = ObjectDetectionAnnotation.from_objects(
        [make_annotated_object(), make_annotated_object(label=1)],
        label_map=["cat", "dog"],
    )
    data = ann.to_dict()
    restored = ObjectDetectionAnnotation.from_dict(data)
    assert restored.to_dict() == data


def test_object_detection_annotation_no_objects() -> None:
    ann = ObjectDetectionAnnotation(label_map=["a"])
    assert ann.to_objects() == []
    data = ann.to_dict()
    assert data["labels"] is None
    restored = ObjectDetectionAnnotation.from_dict(data)
    assert restored.labels is None


def test_layout_analysis_annotation_type() -> None:
    ann = LayoutAnalysisAnnotation(label_map=["a"])
    assert ann.type == AnnotationType.layout_analysis.value
    assert ann.to_dict()["type"] == "layout_analysis"


@pytest.mark.parametrize(
    "maker",
    [
        lambda: make_classification_annotation(),
        lambda: make_entity_labeling_annotation(),
        lambda: make_question_answering_annotation(),
        lambda: make_object_detection_annotation(),
        lambda: LayoutAnalysisAnnotation(label_map=["a"]),
    ],
)
def test_annotation_to_dict_from_dict_roundtrip(maker) -> None:
    ann = maker()
    data = ann.to_dict()
    restored = type(ann).from_dict(data)
    assert restored.to_dict() == data
