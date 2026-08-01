# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._common import (
        ConfigType,
        DatasetSplitType,
        GANStage,
        ModelType,
        OCRType,
        TaskType,
    )
    from ._data_instance._base import BaseDataInstance
    from ._data_instance._document_instance import DocumentInstance
    from ._data_instance._image_instance import ImageInstance
    from ._datasets import (
        DatasetLabels,
        DatasetMetadata,
        DatasetShardInfo,
        SplitConfig,
        SplitInfo,
    )
    from ._generic._doc_content import DocumentContent
    from ._generic._elements import ElementArray, OCRLevel
    from ._generic._annotated_object import AnnotatedObject
    from ._generic._bounding_box import BoundingBoxMode
    from ._generic._annotations import (
        Annotation,
        EntityLabelingAnnotation,
        ClassificationAnnotation,
        LayoutAnalysisAnnotation,
        QuestionAnsweringAnnotation,
        ObjectDetectionAnnotation,
        AnnotationType,
    )
    from ._generic._image import Image
    from ._generic._qa_pair import QAPair
    from ._utilities._repr import RepresentationMixin

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
        "_data_instance._base": ["BaseDataInstance"],
        "_data_instance._document_instance": ["DocumentInstance"],
        "_data_instance._image_instance": ["ImageInstance"],
        "_datasets": [
            "DatasetLabels",
            "DatasetMetadata",
            "DatasetShardInfo",
            "SplitConfig",
            "SplitInfo",
        ],
        "_generic._doc_content": ["DocumentContent"],
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
            "AnnotationType",
        ],
        "_generic._image": ["Image"],
        "_generic._qa_pair": ["QAPair"],
        "_utilities._repr": ["RepresentationMixin"],
    },
)
