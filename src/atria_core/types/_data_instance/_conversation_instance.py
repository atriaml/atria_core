from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._data_instance._base import DataInstance, Metadata
from atria_core.types._generic._conversation_turn import (
    ConversationRole,
    ConversationTurn,
)


@dataclass(frozen=True, repr=False)
class ConversationInstance(DataInstance):
    """An ordered sequence of conversation turns -- for multi-turn dialogue
    datasets (e.g. chat logs, role-played conversations)."""

    turns: list[ConversationTurn]

    def load(self) -> ConversationInstance:
        return self

    def to_conversation_seed(self) -> ConversationInstance:
        return ConversationInstance(
            sample_id=self.sample_id,
            metadata=self.metadata,
            turns=[turn for turn in self.turns if turn.role == ConversationRole.user],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metadata": self.metadata.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationInstance:
        return cls(
            sample_id=data["sample_id"],
            metadata=Metadata.from_dict(data.get("metadata", {})),
            turns=[ConversationTurn.from_dict(turn) for turn in data["turns"]],
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )


@dataclass(frozen=True, repr=False)
class ConversationSeed(ConversationInstance):
    """Seeds for a conversations which act as user prompts for any conversation"""

    def __post_init__(self) -> None:
        if not all(turn.role == ConversationRole.user for turn in self.turns):
            raise ValueError("All turns in a ConversationSeed must have role 'user'.")
