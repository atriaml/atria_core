from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._data_instance._base import DataInstance
from atria_core.types._generic._conversation_turn import ConversationTurn


@dataclass(frozen=True, repr=False)
class ConversationInstance(DataInstance):
    """An ordered sequence of conversation turns -- for multi-turn dialogue
    datasets (e.g. chat logs, role-played conversations)."""

    turns: list[ConversationTurn]

    def load(self) -> ConversationInstance:
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationInstance:
        return cls(
            sample_id=data["sample_id"],
            turns=[ConversationTurn.from_dict(turn) for turn in data["turns"]],
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )
