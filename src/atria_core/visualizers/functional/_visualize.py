from __future__ import annotations

from pathlib import Path
from typing import Any

from atria_core.types import (
    AnnotationType,
    BaseDataInstance,
    DocumentInstance,
    ImageInstance,
)


def output_name(instance: BaseDataInstance) -> str:
    classification_annotation = instance.get_annotation_by_type(
        AnnotationType.classification
    )
    if classification_annotation is None:
        return instance.sample_id
    return f"{instance.sample_id}_label={classification_annotation.label_name}"


def visualize(instance: BaseDataInstance, output_dir: str, **kwargs: Any) -> Path:
    from atria_core.visualizers.functional._document_instance import (
        visualize_document_instance,
    )
    from atria_core.visualizers.functional._image_instance import (
        visualize_image_instance,
    )

    if isinstance(instance, ImageInstance):
        return visualize_image_instance(instance, output_dir, **kwargs)
    if isinstance(instance, DocumentInstance):
        return visualize_document_instance(instance, output_dir, **kwargs)
    raise TypeError(f"No visualizer for instance type: {type(instance).__name__}")
