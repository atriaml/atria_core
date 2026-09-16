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
    alternative_answers: list[str] = field(default_factory=list[str])

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


@dataclass(frozen=True, repr=False)
class MultiPageQAPair(BaseDataModel):
    """A question-answer pair over a multi-page document, naming which pages
    support the answer and, when the answer is computed rather than quoted,
    the arithmetic expression used to derive it."""

    id: int
    question_text: str
    answer_text: str
    evidence_pages: list[int] = field(default_factory=list[int])
    arithmetic_expression: str | None = None
    alternative_answers: list[str] = field(default_factory=list[str])
    evidence_sources: list[str] = field(default_factory=list[str])
    answer_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question_text": self.question_text,
            "answer_text": self.answer_text,
            "evidence_pages": self.evidence_pages,
            "arithmetic_expression": self.arithmetic_expression,
            "alternative_answers": self.alternative_answers,
            "evidence_sources": self.evidence_sources,
            "answer_format": self.answer_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiPageQAPair:
        return cls(
            id=data["id"],
            question_text=data["question_text"],
            answer_text=data["answer_text"],
            evidence_pages=list(data.get("evidence_pages", [])),
            arithmetic_expression=data.get("arithmetic_expression"),
            alternative_answers=list(data.get("alternative_answers", [])),
            evidence_sources=list(data.get("evidence_sources", [])),
            answer_format=data.get("answer_format"),
        )
