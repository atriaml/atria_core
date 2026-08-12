# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._cached_dataset import CachedDataset
    from ._cacher import Cacher, PreprocessTransform
    from ._common import DatasetLoadingMode, FileStorageType
    from ._dataset import Dataset, DatasetConfig
    from ._registry import datasets
    from ._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
    from ._split_iterators import (
        Compose,
        IndexableSplitIterator,
        IterableSplitIterator,
    )
    from ._snapshot import DatasetSnapshot
    from ._snapshot_store import DatasetSnapshotStore

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_cached_dataset": ["CachedDataset"],
        "_cacher": ["Cacher", "PreprocessTransform"],
        "_common": [
            "DatasetLoadingMode",
            "FileStorageType",
        ],
        "_dataset": [
            "Dataset",
            "DatasetConfig",
        ],
        "_registry": ["datasets"],
        "_hf_dataset": [
            "HuggingfaceDataset",
            "HuggingfaceDatasetConfig",
        ],
        "_split_iterators": [
            "Compose",
            "IndexableSplitIterator",
            "IterableSplitIterator",
        ],
        "_snapshot": ["DatasetSnapshot"],
        "_snapshot_store": ["DatasetSnapshotStore"],
    },
)
