# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._builder import DatasetBuilder as DatasetBuilder
    from ._cached_dataset import (
        CachedDataset as CachedDataset,
    )
    from ._cacher import Cacher as Cacher, PreprocessTransform as PreprocessTransform
    from ._common import (
        DatasetLoadingMode as DatasetLoadingMode,
        FileStorageType as FileStorageType,
    )
    from ._dataset import Dataset as Dataset, DatasetConfig as DatasetConfig
    from ._download._download_manager import (
        AtriaDownloadManager as AtriaDownloadManager,
        UrlSpec as UrlSpec,
    )
    from ._hf_dataset import (
        HuggingfaceDataset as HuggingfaceDataset,
        HuggingfaceDatasetConfig as HuggingfaceDatasetConfig,
    )
    from ._registry import DatasetRegistry as DatasetRegistry, datasets as datasets
    from ._split_iterators import (
        Compose as Compose,
        ConcatSplitIterator as ConcatSplitIterator,
        IndexableSplitIterator as IndexableSplitIterator,
        IterableSplitIterator as IterableSplitIterator,
    )
    from ._snapshot import DatasetSnapshot as DatasetSnapshot
    from ._snapshot_store import DatasetSnapshotStore as DatasetSnapshotStore

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_builder": ["DatasetBuilder"],
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
        "_download._download_manager": ["AtriaDownloadManager", "UrlSpec"],
        "_hf_dataset": [
            "HuggingfaceDataset",
            "HuggingfaceDatasetConfig",
        ],
        "_registry": [
            "DatasetRegistry",
            "datasets",
        ],
        "_split_iterators": [
            "Compose",
            "ConcatSplitIterator",
            "IndexableSplitIterator",
            "IterableSplitIterator",
        ],
        "_snapshot": ["DatasetSnapshot"],
        "_snapshot_store": ["DatasetSnapshotStore"],
    },
)
