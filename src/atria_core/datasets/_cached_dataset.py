from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Generic

import yaml

from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
from atria_core.datasets._dataset import Dataset, DatasetConfig, T_BaseDataInstance
from atria_core.datasets._snapshot import DatasetSnapshot
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.logger import get_logger
from atria_core.registry import ModuleConfig
from atria_core.types import DatasetMetadata, DatasetSplitType

logger = get_logger(__name__)


class _SafeTupleLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(
    loader: yaml.SafeLoader, node: yaml.SequenceNode
) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


_SafeTupleLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)


class CachedDataset(
    Dataset[DatasetConfig, T_BaseDataInstance], Generic[T_BaseDataInstance]
):
    """A dataset read from an on-disk cache directory.

    Its sample class and config are reconstructed from the snapshot stored
    alongside the records."""

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
        self._data_model_cls: type[T_BaseDataInstance] = getattr(module, class_name)

        config_data = self._snapshot.config
        if not config_data:
            config_path = self._path / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        config = ModuleConfig.from_dict(config_data)

        super().__init__(config=config, data_dir=str(self._path))  # type: ignore[arg-type]

    @property
    def data_dir(self) -> Path:
        """The cache directory this dataset was opened from."""
        return self._path

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
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
        if self._snapshot.metadata:
            return self._snapshot.dataset_metadata
        metadata_path = self._path / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return DatasetMetadata.from_dict(yaml.safe_load(f))
        return DatasetMetadata()

    def _build_input_transform(
        self, **kwargs: Any
    ) -> Callable[[Any], T_BaseDataInstance]:
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
