from __future__ import annotations

from atria_core.types._data_instance._conversation_instance import ConversationInstance
from atria_core.types._generic._annotations import AnnotationType
from atria_core.types._generic._conversation_turn import (
    ConversationRole,
    ConversationTurn,
)
from tests.types.builders import make_classification_annotation


def _turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(role=ConversationRole.user, text="hello"),
        ConversationTurn(role=ConversationRole.assistant, text="hi there"),
    ]


def test_construct_and_repr() -> None:
    instance = ConversationInstance(sample_id="s1", turns=_turns())
    assert repr(instance)
    assert instance.turns[0].role == ConversationRole.user
    assert instance.turns[1].text == "hi there"


def test_equality() -> None:
    a = ConversationInstance(sample_id="s1", turns=_turns())
    b = ConversationInstance(sample_id="s1", turns=_turns())
    c = ConversationInstance(sample_id="s2", turns=_turns())
    assert a == b
    assert a != c


def test_round_trips_through_dict() -> None:
    instance = ConversationInstance(sample_id="s1", turns=_turns())
    restored = ConversationInstance.from_dict(instance.to_dict())
    assert restored == instance


def test_inherits_base_data_instance_behavior() -> None:
    ann = make_classification_annotation()
    instance = ConversationInstance(sample_id="s1", turns=_turns()).add_annotation(ann)
    assert instance.key == "s1"
    assert instance.get_annotation_by_type(AnnotationType.classification) is ann
