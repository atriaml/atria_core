from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from atria_core.serialization._artifact_store import ArtifactStore
from atria_core.types import (
    BaseDataInstance,
    ImageInstance,
    MultiPageDocumentInstance,
    PdfPage,
    SinglePageDocumentInstance,
)

_IMAGE_INSTANCE = "image_instance"
_SINGLE_PAGE_DOCUMENT_INSTANCE = "single_page_document_instance"
_MULTI_PAGE_DOCUMENT_INSTANCE = "multi_page_document_instance"


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
        elif isinstance(instance, SinglePageDocumentInstance):
            instance = RowCodec._materialize_single_page(key, instance, store)
            row_type = _SINGLE_PAGE_DOCUMENT_INSTANCE
        elif isinstance(instance, MultiPageDocumentInstance):
            instance = replace(
                instance, source_path=store.materialize_pdf(key, instance.source_path)
            )
            row_type = _MULTI_PAGE_DOCUMENT_INSTANCE
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
        if row["type"] == _SINGLE_PAGE_DOCUMENT_INSTANCE:
            return SinglePageDocumentInstance.from_dict(data)
        if row["type"] == _MULTI_PAGE_DOCUMENT_INSTANCE:
            return MultiPageDocumentInstance.from_dict(data)
        raise ValueError(f"Unknown instance type: {row['type']!r}")

    @staticmethod
    def _materialize_single_page(
        key: str, instance: SinglePageDocumentInstance, store: ArtifactStore
    ) -> SinglePageDocumentInstance:
        if isinstance(instance.visual, PdfPage):
            assert instance.visual.file_path is not None
            copied_path = store.materialize_pdf(key, instance.visual.file_path)
            return replace(
                instance, visual=replace(instance.visual, file_path=copied_path)
            )
        return replace(instance, visual=store.materialize_image(key, instance.visual))
