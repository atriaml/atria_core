from __future__ import annotations

import json


class AnswerSpan:
    def __init__(self, start: int, end: int, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}

    def __repr__(self) -> str:
        return f"AnswerSpan(start={self.start}, end={self.end}, text={self.text!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnswerSpan):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and self.text == other.text
        )


class QAPair:
    def __init__(
        self, id: int, question_text: str, answer_spans: list[AnswerSpan] | str
    ) -> None:
        self.id = id
        self.question_text = question_text
        self.answer_spans = self._parse_answer_spans(answer_spans)

    @staticmethod
    def _parse_answer_spans(value: list[AnswerSpan] | str) -> list[AnswerSpan]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return [
            AnswerSpan(**item) if isinstance(item, dict) else item for item in value
        ]

    @property
    def answers(self) -> list[str]:
        if not self.answer_spans:
            return []
        return [span.text for span in self.answer_spans]

    def serialize_answer_spans(self) -> str:
        return json.dumps([span.to_dict() for span in self.answer_spans])

    def __repr__(self) -> str:
        return (
            f"QAPair(id={self.id}, question_text={self.question_text!r}, "
            f"answer_spans={self.answer_spans!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QAPair):
            return NotImplemented
        return (
            self.id == other.id
            and self.question_text == other.question_text
            and self.answer_spans == other.answer_spans
        )
