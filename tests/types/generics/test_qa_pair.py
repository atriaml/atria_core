from __future__ import annotations

from atria_core.types._generic._qa_pair import QAPair
from tests.types.builders import make_qa_pair


def test_construct_with_span() -> None:
    qa = make_qa_pair(start=0, end=4)
    assert qa.start == 0
    assert qa.end == 4


def test_construct_without_span() -> None:
    qa = QAPair(id=1, question_text="why?", answer_text="because")
    assert qa.start is None
    assert qa.end is None


def test_equality() -> None:
    a = make_qa_pair()
    b = make_qa_pair()
    c = make_qa_pair(answer_text="different")
    assert a == b
    assert a != c


def test_to_dict_from_dict_roundtrip() -> None:
    qa = make_qa_pair()
    data = qa.to_dict()
    restored = QAPair.from_dict(data)
    assert restored == qa


def test_to_dict_from_dict_roundtrip_without_span() -> None:
    qa = QAPair(id=2, question_text="q", answer_text="a")
    restored = QAPair.from_dict(qa.to_dict())
    assert restored == qa


def test_alternative_answers_defaults_to_empty() -> None:
    qa = QAPair(id=3, question_text="q", answer_text="a")
    assert qa.alternative_answers == []


def test_to_dict_from_dict_roundtrip_with_alternative_answers() -> None:
    qa = QAPair(
        id=4,
        question_text="q",
        answer_text="a",
        alternative_answers=["a", "alternative a", "another a"],
    )
    restored = QAPair.from_dict(qa.to_dict())
    assert restored == qa
    assert restored.alternative_answers == ["a", "alternative a", "another a"]
