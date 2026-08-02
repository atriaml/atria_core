# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._cached_dataset import CachedDataset as CachedDataset
    from ._cacher import Cacher as Cacher
    from ._common import DatasetLoadingMode as DatasetLoadingMode
    from ._common import FileStorageType as FileStorageType
    from ._dataset import Dataset as Dataset
    from ._dataset import DatasetConfig as DatasetConfig
    from ._dataset_builders import PreprocessTransform as PreprocessTransform
    from ._exceptions import ConfigurationNotFoundError as ConfigurationNotFoundError
    from ._exceptions import SplitNotFoundError as SplitNotFoundError
    from ._hf_dataset import HuggingfaceDataset as HuggingfaceDataset
    from ._hf_dataset import HuggingfaceDatasetConfig as HuggingfaceDatasetConfig
    from ._split_iterators import Compose as Compose
    from ._split_iterators import IndexableSplitIterator as IndexableSplitIterator
    from ._split_iterators import IterableSplitIterator as IterableSplitIterator
    from ._split_iterators import compose as compose

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_cached_dataset": ["CachedDataset"],
        "_cacher": ["Cacher"],
        "_common": [
            "DatasetLoadingMode",
            "FileStorageType",
        ],
        "_dataset": [
            "Dataset",
            "DatasetConfig",
        ],
        "_dataset_builders": ["PreprocessTransform"],
        "_exceptions": ["ConfigurationNotFoundError", "SplitNotFoundError"],
        "_hf_dataset": [
            "HuggingfaceDataset",
            "HuggingfaceDatasetConfig",
        ],
        "_split_iterators": [
            "Compose",
            "IndexableSplitIterator",
            "IterableSplitIterator",
            "compose",
        ],
    },
)
