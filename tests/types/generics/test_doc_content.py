from __future__ import annotations

from atria_core.types._generic._doc_content import DocumentContent
from tests.types.builders import make_document_content, make_element_array


def test_derives_text_from_elements() -> None:
    content = make_document_content(
        elements=make_element_array(texts=["hello", "world"])
    )
    assert content.text == "hello world"


def test_explicit_text_not_overridden() -> None:
    content = DocumentContent(_text="explicit", elements=make_element_array())
    assert content.text == "explicit"


def test_no_elements_no_text() -> None:
    content = DocumentContent()
    assert content.text is None
    assert content.elements is None


def test_to_dict_from_dict_roundtrip() -> None:
    content = make_document_content(
        elements=make_element_array(texts=["hello", "world"])
    )
    data = content.to_dict()
    restored = DocumentContent.from_dict(data)
    assert restored.to_dict() == data


def test_to_dict_handles_no_elements() -> None:
    content = DocumentContent(_text="just text")
    data = content.to_dict()
    assert data == {"text": "just text", "elements": None}
    restored = DocumentContent.from_dict(data)
    assert restored.text == "just text"
    assert restored.elements is None
