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
    system = "system"


@dataclass(frozen=True, repr=False)
class ConversationTurn(BaseDataModel):
    """One message in a conversation: who sent it, and what it says."""

    role: ConversationRole
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        return cls(role=ConversationRole(data["role"]), text=data["text"])
