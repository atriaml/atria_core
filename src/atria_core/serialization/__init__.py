# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._artifact_store import ArtifactStore
    from ._dataset_reader import DatasetReader
    from ._dataset_writer import DatasetWriter
    from ._row_codec import RowCodec

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_artifact_store": ["ArtifactStore"],
        "_dataset_reader": ["DatasetReader"],
        "_dataset_writer": ["DatasetWriter"],
        "_row_codec": ["RowCodec"],
    },
)
