# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    # Self-aliased (`as Name`) so mypy's --no-implicit-reexport (part of
    # `strict`) treats these as explicit re-exports -- required for any
    # `from atria_core.types import X` done outside this package, since
    # `__all__` below is computed at runtime by lazy_loader and isn't
    # visible to mypy as a literal list.
    from ._common import (
        ConfigType as ConfigType,
        DatasetSplitType as DatasetSplitType,
        GANStage as GANStage,
        ModelType as ModelType,
        OCRType as OCRType,
        TaskType as TaskType,
    )
    from ._data_instance._base import DataInstance as DataInstance
    from ._data_instance._document_instance import (
        DocumentInstance as DocumentInstance,
        MultiPageDocumentInstance as MultiPageDocumentInstance,
        SinglePageDocumentInstance as SinglePageDocumentInstance,
    )
    from ._data_instance._image_instance import ImageInstance as ImageInstance
    from ._data_instance._text_instance import TextInstance as TextInstance
    from ._datasets import (
        DatasetLabels as DatasetLabels,
        DatasetMetadata as DatasetMetadata,
        DatasetShardInfo as DatasetShardInfo,
        SplitConfig as SplitConfig,
        SplitInfo as SplitInfo,
    )
    from ._generic._doc_content import DocumentContent as DocumentContent
    from ._generic._documents import PdfPage as PdfPage
    from ._generic._elements import ElementArray as ElementArray, OCRLevel as OCRLevel
    from ._generic._annotated_object import AnnotatedObject as AnnotatedObject
    from ._generic._bounding_box import BoundingBoxMode as BoundingBoxMode
    from ._generic._annotations import (
        Annotation as Annotation,
        EntityLabelingAnnotation as EntityLabelingAnnotation,
        ClassificationAnnotation as ClassificationAnnotation,
        LayoutAnalysisAnnotation as LayoutAnalysisAnnotation,
        QuestionAnsweringAnnotation as QuestionAnsweringAnnotation,
        ObjectDetectionAnnotation as ObjectDetectionAnnotation,
        TranscriptionAnnotation as TranscriptionAnnotation,
        OCRAnnotation as OCRAnnotation,
        AnnotationType as AnnotationType,
    )
    from ._generic._image import Image as Image
    from ._generic._qa_pair import QAPair as QAPair
    from ._utilities._repr import RepresentationMixin as RepresentationMixin
    from ._utilities._url_fetchers import (
        LocalResourceLoader as LocalResourceLoader,
        RemoteResourceLoader as RemoteResourceLoader,
        ResourceLoader as ResourceLoader,
    )

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_common": [
            "ConfigType",
            "DatasetSplitType",
            "GANStage",
            "ModelType",
            "OCRType",
            "TaskType",
        ],
        "_data_instance._base": ["DataInstance"],
        "_data_instance._document_instance": [
            "DocumentInstance",
            "MultiPageDocumentInstance",
            "SinglePageDocumentInstance",
        ],
        "_data_instance._image_instance": ["ImageInstance"],
        "_data_instance._text_instance": ["TextInstance"],
        "_datasets": [
            "DatasetLabels",
            "DatasetMetadata",
            "DatasetShardInfo",
            "SplitConfig",
            "SplitInfo",
        ],
        "_generic._doc_content": ["DocumentContent"],
        "_generic._documents": ["PdfPage"],
        "_generic._elements": ["ElementArray", "OCRLevel"],
        "_generic._annotated_object": ["AnnotatedObject"],
        "_generic._bounding_box": ["BoundingBoxMode"],
        "_generic._annotations": [
            "Annotation",
            "EntityLabelingAnnotation",
            "ClassificationAnnotation",
            "LayoutAnalysisAnnotation",
            "QuestionAnsweringAnnotation",
            "ObjectDetectionAnnotation",
            "TranscriptionAnnotation",
            "OCRAnnotation",
            "AnnotationType",
        ],
        "_generic._image": ["Image"],
        "_generic._qa_pair": ["QAPair"],
        "_utilities._repr": ["RepresentationMixin"],
        "_utilities._url_fetchers": [
            "LocalResourceLoader",
            "RemoteResourceLoader",
            "ResourceLoader",
        ],
    },
)
