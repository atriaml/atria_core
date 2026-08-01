from __future__ import annotations

import shutil

import pytest
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
    annotated_object = AnnotatedObject(label=1, bbox=bbox)
    elements = ElementArray.from_words(["hello"], [bbox])
    doc_content = DocumentContent(elements=elements)
    qa_pair = QAPair(id=0, question_text="what?", answer_text="this", start=0, end=4)

    for obj in [bbox, annotated_object, elements, doc_content, qa_pair]:
        assert repr(obj)
        assert obj == obj


def test_construct_repr_eq_instances() -> None:
    from atria_core.types import Image, ImageInstance

    image = Image(PILImage.new("RGB", (4, 4)))
    instance = ImageInstance(sample_id="s1", image=image)

    assert repr(instance)
    assert instance == instance


def test_content_extractor_recursion() -> None:
    from atria_core.types._extractors._base import ContentExtractor
    from atria_core.types._generic._doc_content import DocumentContent
    from atria_core.types._generic._documents import SinglePageDocument

    class StubExtractor(ContentExtractor):
        def _extract(self, image: PILImage.Image) -> DocumentContent:
            return DocumentContent(text="stub")

    doc = SinglePageDocument.from_image(PILImage.new("RGB", (4, 4)))
    extracted = doc.extract_content(StubExtractor())

    assert isinstance(extracted, SinglePageDocument)
    assert extracted.content is not None


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)
def test_tesseract_extractor_end_to_end() -> None:
    from atria_core.types._extractors._tesseract import TesseractExtractorConfig
    from atria_core.types._generic._documents import SinglePageDocument

    image = PILImage.new("RGB", (100, 40), color="white")
    doc = SinglePageDocument.from_image(image)
    extracted = doc.extract_content(TesseractExtractorConfig())

    assert isinstance(extracted, SinglePageDocument)
    assert extracted.content is not None
