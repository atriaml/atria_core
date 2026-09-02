from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from atria_core.types._base_data_model import BaseDataModel


class ConversationRole(str, enum.Enum):
    """Fixed role vocabulary every ConversationInstance producer maps its
    own source roles onto, so consumers can filter by role without knowing
    which dataset a conversation came from."""

    user = "user"
    assistant = "assistant"
    thinking = "thinking"
    tool = "tool"
    system = "system"


@dataclass(frozen=True, repr=False)
class ConversationTurn(BaseDataModel):
    """One message in a conversation: who sent it, and what it says."""

    role: ConversationRole | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        if self.role is None:
            return {"text": self.text}
        return {
            "role": self.role.value,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        role = data.get("role", None)
        return cls(
            role=ConversationRole(role) if role is not None else None, text=data["text"]
        )
