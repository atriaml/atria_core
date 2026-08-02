# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._cached_dataset import CachedDataset as CachedDataset
    from ._common import DatasetConfig as DatasetConfig
    from ._common import DatasetLoadingMode as DatasetLoadingMode
    from ._common import FileStorageType as FileStorageType
    from ._common import HuggingfaceDatasetConfig as HuggingfaceDatasetConfig
    from ._dataset import Dataset as Dataset
    from ._dataset import DatasetInputTransform as DatasetInputTransform
    from ._dataset import DocumentDataset as DocumentDataset
    from ._dataset import ImageDataset as ImageDataset
    from ._dataset_builders import ComposedTransform as ComposedTransform
    from ._dataset_builders import PreprocessTransform as PreprocessTransform
    from ._exceptions import ConfigurationNotFoundError as ConfigurationNotFoundError
    from ._exceptions import SplitNotFoundError as SplitNotFoundError
    from ._hf_dataset import HuggingfaceDataset as HuggingfaceDataset
    from ._hf_dataset import HuggingfaceDocumentDataset as HuggingfaceDocumentDataset
    from ._hf_dataset import HuggingfaceImageDataset as HuggingfaceImageDataset
    from ._split_iterators import HFSplitIterator as HFSplitIterator
    from ._split_iterators import InstanceTransform as InstanceTransform
    from ._split_iterators import SplitIterator as SplitIterator

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_cached_dataset": ["CachedDataset"],
        "_common": [
            "DatasetConfig",
            "DatasetLoadingMode",
            "FileStorageType",
            "HuggingfaceDatasetConfig",
        ],
        "_dataset": [
            "Dataset",
            "DatasetInputTransform",
            "DocumentDataset",
            "ImageDataset",
        ],
        "_dataset_builders": ["ComposedTransform", "PreprocessTransform"],
        "_exceptions": ["ConfigurationNotFoundError", "SplitNotFoundError"],
        "_hf_dataset": [
            "HuggingfaceDataset",
            "HuggingfaceDocumentDataset",
            "HuggingfaceImageDataset",
        ],
        "_split_iterators": ["HFSplitIterator", "InstanceTransform", "SplitIterator"],
    },
)
