from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from atria_core.serialization._artifact_store import ArtifactStore
from atria_core.types import (
    BaseDataInstance,
    DocumentInstance,
    ImageInstance,
    MultiPageDocument,
    SinglePageDocument,
)

_IMAGE_INSTANCE = "image_instance"
_DOCUMENT_INSTANCE = "document_instance"


class RowCodec:
    """Converts one BaseDataInstance to/from a flat parquet row
    (sample_id, type, data_json), materializing any binary content (images,
    PDFs) into the given ArtifactStore along the way."""

    @staticmethod
    def to_row(instance: BaseDataInstance, store: ArtifactStore) -> dict[str, Any]:
        key = instance.key

        if isinstance(instance, ImageInstance):
            instance = replace(
                instance, image=store.materialize_image(key, instance.image)
            )
            row_type = _IMAGE_INSTANCE
        elif isinstance(instance, DocumentInstance):
            instance = replace(
                instance, document=RowCodec._materialize_document(key, instance.document, store)
            )
            row_type = _DOCUMENT_INSTANCE
        else:
            raise TypeError(f"Unsupported instance type: {type(instance).__name__}")

        return {
            "sample_id": instance.sample_id,
            "type": row_type,
            "data_json": json.dumps(instance.to_dict()),
        }

    @staticmethod
    def from_row(row: dict[str, Any]) -> BaseDataInstance:
        data = json.loads(row["data_json"])
        if row["type"] == _IMAGE_INSTANCE:
            return ImageInstance.from_dict(data)
        if row["type"] == _DOCUMENT_INSTANCE:
            return DocumentInstance.from_dict(data)
        raise ValueError(f"Unknown instance type: {row['type']!r}")

    @staticmethod
    def _materialize_document(
        key: str,
        document: SinglePageDocument | MultiPageDocument,
        store: ArtifactStore,
    ) -> SinglePageDocument | MultiPageDocument:
        if isinstance(document, MultiPageDocument):
            return replace(document, source_path=store.materialize_pdf(key, document.source_path))

        path = store.materialize_page_image(key, document.image)
        return replace(document, source_path=path)
