from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from atria_core.datasets._common import FileStorageType
from atria_core.datasets._dataset import Dataset, DatasetConfig
from atria_core.datasets._snapshot import DatasetSnapshot
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.logger import get_logger
from atria_core.types import DatasetMetadata, DatasetSplitType
from atria_core.types._data_instance._base import DataInstance

logger = get_logger(__name__)


class _CachedDatasetConfig(DatasetConfig):
    """The params recorded in a cache's snapshot, carried as plain data.

    A cache is described entirely by its snapshot, and reading it needs no
    knowledge of the dataset that produced it -- whose config class may not
    even be importable. So this declares no fields of its own and accepts
    whatever params the snapshot holds, rather than rebuilding the original
    config class.
    """

    model_config = ConfigDict(frozen=True, extra="allow")


class CachedDataset[T_DataInstance: DataInstance = DataInstance](
    Dataset[T_DataInstance, _CachedDatasetConfig]
):
    """A dataset read from an on-disk cache directory.

    Its sample class comes from the snapshot stored alongside the records, and
    its config carries that snapshot's params verbatim."""

    __module_name__ = "cached"

    def __init__(self, path: Path | str) -> None:
        """Open a cache directory.

        Args:
            path: Directory holding a cached-dataset snapshot.

        Raises:
            ValueError: If the snapshot there is not a cached-dataset snapshot.
        """
        self._path = Path(path)

        self._snapshot = DatasetSnapshot.load(self._path)
        if not self._snapshot.is_cached:
            raise ValueError(
                f"Snapshot at {self._path} is a {self._snapshot.snapshot_kind} snapshot, not a cached dataset snapshot."
            )

        fqn = self._snapshot.data_model
        assert fqn is not None
        module_name, class_name = fqn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self._data_model_cls: type[T_DataInstance] = getattr(module, class_name)

        super().__init__(
            config=_CachedDatasetConfig.from_dict(self._snapshot.config),
            data_dir=str(self._path),
        )

    @property
    def data_dir(self) -> Path:
        """The cache directory this dataset was opened from."""
        return self._path

    @property
    def data_model(self) -> type[T_DataInstance]:
        """Sample class the cached records deserialize into."""
        return self._data_model_cls

    @property
    def dataset_class_name(self) -> str:
        """Name of the Dataset class this cache was built from."""
        return self._snapshot.dataset_class_name

    @property
    def storage_type(self) -> FileStorageType:
        """On-disk format these records are stored in."""
        assert self._snapshot.storage_type is not None
        return self._snapshot.storage_type

    @property
    def config_name(self) -> str:
        """Cache directory name, which encodes class and config hash."""
        assert self._snapshot.config_name is not None
        return self._snapshot.config_name

    @property
    def config_hash(self) -> str:
        """Hash of the config of the dataset this cache was built from."""
        assert self._snapshot.config_hash is not None
        return self._snapshot.config_hash

    @property
    def snapshot(self) -> DatasetSnapshot:
        """The snapshot describing this cache."""
        return self._snapshot

    @property
    def dataset_stage(self) -> str:
        """Whether these records are raw or transformed."""
        return self._snapshot.dataset_stage

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        return {}  # nothing to fetch, already cached

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return self._storage_manager().get_splits()

    def _metadata(self) -> DatasetMetadata:
        return self._snapshot.dataset_metadata

    def _build_input_transform(self, **kwargs: Any) -> Callable[[Any], T_DataInstance]:
        return self.data_model.from_dict  # type: ignore[return-value]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Sequence[Any] | Iterable[Any]:
        return self._storage_manager().read_split(split)

    def _storage_manager(self) -> StorageManager:
        return StorageManager.create(
            self.storage_type,
            data_dir=str(self._path.parent),
            storage_dir=str(self._path.parent),
            config_name=self._path.name,
            num_processes=1,
        )

    def _persist_snapshot(self) -> None:
        return
