from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atria_core.types._base_data_model import BaseDataModel


@dataclass(frozen=True, repr=False)
class QAPair(BaseDataModel):
    id: int
    question_text: str
    answer_text: str
    start: int | None = None
    end: int | None = None
    #: Every acceptable gold answer string, when a dataset provides more than
    #: one (e.g. SQuAD). Empty for datasets with a single gold answer --
    #: consumers should fall back to `answer_text` in that case.
    alternative_answers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question_text": self.question_text,
            "answer_text": self.answer_text,
            "start": self.start,
            "end": self.end,
            "alternative_answers": self.alternative_answers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QAPair:
        return cls(
            id=data["id"],
            question_text=data["question_text"],
            answer_text=data["answer_text"],
            start=data.get("start"),
            end=data.get("end"),
            alternative_answers=list(data.get("alternative_answers", [])),
        )
