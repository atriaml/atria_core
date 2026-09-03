from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._data_instance._base import DataInstance
from atria_core.types._generic._conversation_turn import (
    ConversationItem,
    ConversationRole,
    ConversationTurn,
    ToolCall,
    ToolResult,
)


@dataclass(frozen=True, repr=False)
class ConversationInstance(DataInstance):
    """An ordered sequence of conversation turns -- for multi-turn dialogue
    datasets (e.g. chat logs, role-played conversations)."""

    turns: list[ConversationTurn]

    def load(self) -> ConversationInstance:
        return self

    def to_conversation_seed(self) -> ConversationSeed:
        return ConversationSeed(
            sample_id=self.sample_id,
            metadata=self.metadata,
            turns=[
                turn
                for turn in self.turns
                if isinstance(turn, ConversationItem)
                and turn.role == ConversationRole.user
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metadata": self.metadata,
            "turns": [
                (
                    {"type": "turn", **turn.to_dict()}
                    if isinstance(turn, ConversationItem)
                    else turn.to_dict()
                )
                for turn in self.turns
            ],
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationInstance:
        turns: list[ConversationTurn] = []

        for turn in data["turns"]:
            turn_type = turn.get("type", "turn")

            if turn_type == "turn":
                turns.append(ConversationItem.from_dict(turn))
            elif turn_type == "tool_call":
                turns.append(ToolCall.from_dict(turn))
            elif turn_type == "tool_result":
                turns.append(ToolResult.from_dict(turn))
            else:
                raise ValueError(f"Unknown conversation item type: {turn_type!r}")

        return cls(
            sample_id=data["sample_id"],
            metadata=data.get("metadata", {}),
            turns=turns,
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )


@dataclass(frozen=True, repr=False)
class ConversationSeed(ConversationInstance):
    """Seeds for a conversations which act as user prompts for any conversation"""

    def __post_init__(self) -> None:
        if not all(turn.role == ConversationRole.user for turn in self.turns):
            raise ValueError("All turns in a ConversationSeed must have role 'user'.")
