from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Any

from atria_core.types._base_data_model import BaseDataModel


class ConversationRole(enum.StrEnum):
    """Fixed role vocabulary every ConversationInstance producer maps its
    own source roles onto, so consumers can filter by role without knowing
    which dataset a conversation came from."""

    user = "user"
    assistant = "assistant"
    thinking = "thinking"
    tool = "tool"
    tool_call = "tool_call"
    system = "system"
    other = "other"


@dataclass(frozen=True, repr=False)
class ConversationItem(BaseDataModel):
    """One message in a conversation: who sent it, and what it says.

    `custom_role` names the turn's role when `role` is `ConversationRole.other`,
    for a producer-specific role outside the fixed vocabulary. It is ignored
    for every other role.
    """

    role: ConversationRole | None
    text: str
    custom_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.role is None:
            return {"text": self.text}
        data: dict[str, Any] = {"role": self.role.value, "text": self.text}
        if self.role is ConversationRole.other and self.custom_role is not None:
            data["custom_role"] = self.custom_role
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationItem:
        role = data.get("role", None)
        return cls(
            role=ConversationRole(role) if role is not None else None,
            text=data["text"],
            custom_role=data.get("custom_role"),
        )


@dataclass(frozen=True, repr=False)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    @property
    def role(self) -> ConversationRole:
        return ConversationRole.tool_call

    @property
    def text(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "arguments": self.arguments,
            },
            ensure_ascii=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=data["name"],
            arguments=data["arguments"],
        )


@dataclass(frozen=True, repr=False)
class ToolResult:
    name: str
    content: str

    @property
    def role(self) -> ConversationRole:
        return ConversationRole.tool

    @property
    def text(self) -> str:
        return self.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "name": self.name,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        return cls(
            name=data["name"],
            content=data["content"],
        )


ConversationTurn = ConversationItem | ToolCall | ToolResult
