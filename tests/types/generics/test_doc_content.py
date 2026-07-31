from __future__ import annotations

import numpy as np

from atria_core.types._generic._doc_content import DocumentContent, TextElement
from tests.types.builders import make_bounding_box, make_text_element


def test_text_element_to_dict_from_dict_roundtrip() -> None:
    te = make_text_element()
    data = te.to_dict()
    restored = TextElement.from_dict(data)
    assert restored.to_dict() == data


def test_text_element_to_dict_handles_none_bbox() -> None:
    te = TextElement(text="x")
    data = te.to_dict()
    assert data["bbox"] is None
    assert data["segment_bbox"] is None
    restored = TextElement.from_dict(data)
    assert restored.to_dict() == data


def test_document_content_derives_text_from_elements() -> None:
    elements = [make_text_element(text="hello"), make_text_element(text="world")]
    content = DocumentContent(text_elements=elements)
    assert content.text == "hello world"


def test_document_content_explicit_text_not_overridden() -> None:
    content = DocumentContent(text="explicit", text_elements=[make_text_element()])
    assert content.text == "explicit"


def test_document_content_list_properties() -> None:
    bbox = make_bounding_box()
    segment_bbox = make_bounding_box(value=[0.0, 0.0, 1.0, 1.0])
    te = make_text_element(text="a", bbox=bbox, segment_bbox=segment_bbox)
    content = DocumentContent(text_elements=[te])

    assert content.text_list == ["a"]
    assert np.array_equal(content.bbox_list[0], bbox)
    assert np.array_equal(content.segment_bbox_list[0], segment_bbox)


def test_document_content_empty_elements() -> None:
    content = DocumentContent(text_elements=None)
    assert content.text_list == []
    assert content.bbox_list == []
    assert content.segment_bbox_list == []
    assert content.serialize_text_elements() is None


def test_document_content_to_dict_from_dict_roundtrip() -> None:
    content = DocumentContent(
        text_elements=[make_text_element(), make_text_element(text="b")]
    )
    data = content.to_dict()
    restored = DocumentContent.from_dict(data)
    assert restored.to_dict() == data


def test_serialize_text_elements_is_valid_json() -> None:
    import json

    content = DocumentContent(text_elements=[make_text_element()])
    serialized = content.serialize_text_elements()
    assert serialized is not None
    parsed = json.loads(serialized)
    assert isinstance(parsed, list)
    assert parsed[0]["text"] == "hello"
