from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._generic._documents import MultiPageDocument, SinglePageDocument


@dataclass(frozen=True, repr=False)
class DocumentInstance(BaseDataInstance):
    document: SinglePageDocument | MultiPageDocument

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "document_type": "multi_page"
            if isinstance(self.document, MultiPageDocument)
            else "single_page",
            "document": self.document.to_dict(),
            "annotations": self._annotations_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentInstance:
        document_cls = (
            MultiPageDocument if data["document_type"] == "multi_page" else SinglePageDocument
        )
        return cls(
            sample_id=data["sample_id"],
            document=document_cls.from_dict(data["document"]),
            _annotations=cls._annotations_from_dict(data.get("annotations")),
        )
