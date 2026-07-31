from __future__ import annotations

from PIL import Image as PILImage

from atria_core.types._data_instance._document_instance import DocumentInstance
from atria_core.types._generic._documents import SinglePageDocument


def test_construct_and_repr() -> None:
    document = SinglePageDocument.from_image(PILImage.new("RGB", (4, 4)))
    instance = DocumentInstance(sample_id="d1", document=document)
    assert repr(instance)
    assert instance.document is document


def test_equality() -> None:
    document = SinglePageDocument.from_image(PILImage.new("RGB", (4, 4)))
    a = DocumentInstance(sample_id="d1", document=document)
    b = DocumentInstance(sample_id="d1", document=document)
    c = DocumentInstance(sample_id="d2", document=document)
    assert a == b
    assert a != c


def test_not_equal_to_unrelated_type() -> None:
    document = SinglePageDocument.from_image(PILImage.new("RGB", (4, 4)))
    instance = DocumentInstance(sample_id="d1", document=document)
    assert instance != document
