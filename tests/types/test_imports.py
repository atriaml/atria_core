from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

IMPORT_CHECK = [
    "ConfigType",
    "DatasetSplitType",
    "GANStage",
    "ModelType",
    "OCRType",
    "TaskType",
    "BaseDataInstance",
    "DocumentInstance",
    "ImageInstance",
    "DatasetLabels",
    "DatasetMetadata",
    "DatasetShardInfo",
    "SplitConfig",
    "SplitInfo",
    "DocumentContent",
    "MultiPageDocumentInstance",
    "SinglePageDocumentInstance",
    "PdfPage",
    "ElementArray",
    "OCRLevel",
    "AnnotatedObject",
    "BoundingBoxMode",
    "Annotation",
    "EntityLabelingAnnotation",
    "ClassificationAnnotation",
    "LayoutAnalysisAnnotation",
    "QuestionAnsweringAnnotation",
    "ObjectDetectionAnnotation",
    "AnnotationType",
    "Image",
    "QAPair",
    "RepresentationMixin",
    "ResourceLoader",
    "LocalResourceLoader",
    "RemoteResourceLoader",
]


def test_imports() -> None:
    import atria_core.types as types

    for name in IMPORT_CHECK:
        assert hasattr(types, name), f"atria_core.types.{name} is not importable"


def test_construct_repr_eq_generic_types() -> None:
    from atria_core.types import (
        AnnotatedObject,
        DocumentContent,
        ElementArray,
        QAPair,
    )

    bbox = (0.1, 0.1, 0.5, 0.5)
    annotated_object = AnnotatedObject(label=1, bbox=np.asarray(bbox, dtype=np.float64))
    elements = ElementArray.from_words(["hello"], [bbox])
    doc_content = DocumentContent(elements=elements)
    qa_pair = QAPair(id=0, question_text="what?", answer_text="this", start=0, end=4)

    for obj in [bbox, annotated_object, elements, doc_content, qa_pair]:
        assert repr(obj)
        assert obj == obj


def test_construct_repr_eq_instances() -> None:
    from atria_core.types import Image, ImageInstance

    image = Image.from_source(PILImage.new("RGB", (4, 4)))
    instance = ImageInstance(sample_id="s1", image=image)

    assert repr(instance)
    assert instance == instance
