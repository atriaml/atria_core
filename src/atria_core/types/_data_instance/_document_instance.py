from __future__ import annotations

from atria_core.types._data_instance._base import (
    BaseDataInstance,
)
from atria_core.types._generic._annotations import Annotation
from atria_core.types._generic._documents import MultiPageDocument, SinglePageDocument


class DocumentInstance(BaseDataInstance):
    def __init__(
        self,
        sample_id: str,
        document: SinglePageDocument | MultiPageDocument,
        annotations: list[Annotation] | None = None,
    ) -> None:
        super().__init__(sample_id=sample_id, annotations=annotations)
        self.document = document

    def __repr__(self) -> str:
        n = len(self.annotations) if self.annotations else 0
        return (
            f"ImageInstance(sample_id={self.sample_id!r}, "
            f"document={self.document!r}, annotations={n})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SinglePageDocument | MultiPageDocument):
            return NotImplemented
        return super().__eq__(other) and self.document == other.document
