from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Generic

import yaml

from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CONFIG_PATH,
    _DEFAULT_ATRIA_DATASETS_METADATA_PATH,
    _DEFAULT_SNAPSHOT_PATH,
)
from atria_core.datasets._dataset import Dataset, DatasetConfig, T_BaseDataInstance
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


class CachedDataset(Dataset[DatasetConfig, T_BaseDataInstance], Generic[T_BaseDataInstance]):
    """A Dataset subclass reading back from an on-disk cache instead of a
    live source -- same shape as HuggingfaceDataset (which reads from a HF
    builder instead of a live source): _download no-ops, _available_splits/
    _build_split_iterator read from the storage manager, and
    _build_input_transform is just data_model.from_dict, since the "raw
    source item" here is a raw dict already sitting on disk."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

        snapshot_file = self._path / _DEFAULT_SNAPSHOT_PATH
        with open(snapshot_file) as f:
            snapshot_data = yaml.load(f, Loader=_SafeTupleLoader)
        assert snapshot_data is not None, f"Snapshot file is empty: {snapshot_file}"
        self._snapshot_data = snapshot_data

        fqn = snapshot_data["data_model"]
        module_name, class_name = fqn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self._data_model_cls: type[T_BaseDataInstance] = getattr(module, class_name)

        config_path = self._path / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
        with open(config_path) as f:
            config = ModuleConfig.from_dict(yaml.safe_load(f))

        super().__init__(config, data_dir=str(self._path))  # type: ignore[arg-type]

    @property
    def data_dir(self) -> Path:
        return self._path

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        return self._data_model_cls

    @property
    def dataset_class_name(self) -> str:
        return str(self._snapshot_data["dataset_class_name"])

    @property
    def storage_type(self) -> FileStorageType:
        return FileStorageType(self._snapshot_data["storage_type"])

    @property
    def config_name(self) -> str:
        return str(self._snapshot_data["config_name"])

    @property
    def config_hash(self) -> str:
        return str(self._snapshot_data["config_hash"])

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        return {}  # nothing to fetch, already cached

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return self._storage_manager().get_splits()

    def _metadata(self) -> DatasetMetadata:
        metadata_path = self._path / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
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
