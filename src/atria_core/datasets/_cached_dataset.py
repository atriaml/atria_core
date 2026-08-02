from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic

import yaml

from atria_core.datasets._common import (
    DatasetConfig,
    FileStorageType,
    T_BaseDataInstance,
)
from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CONFIG_PATH,
    _DEFAULT_ATRIA_DATASETS_METADATA_PATH,
    _DEFAULT_SNAPSHOT_PATH,
)
from atria_core.datasets._exceptions import SplitNotFoundError
from atria_core.datasets._split_iterators import SplitIterator
from atria_core.logger import get_logger
from atria_core.registry import ModuleConfig
from atria_core.types import DatasetMetadata, DatasetSplitType
from atria_core.types._utilities._repr import RepresentationMixin

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


class CachedDataset(RepresentationMixin, Generic[T_BaseDataInstance]):
    """Immutable, file-backed dataset produced by Cacher.cache(). All I/O
    (reading snapshot.yaml/metadata.yaml, building split iterators) happens
    once, inside __init__ -- there is no separate load() mutating self
    afterward."""

    __repr_fields__ = {"data_model", "data_dir", "split_iterators", "metadata"}

    def __init__(
        self,
        path: Path | str,
        *,
        allowed_keys: set[str] | None = None,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> None:
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

        self._metadata_data: DatasetMetadata | None = None
        metadata_path = self._path / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata_data = DatasetMetadata.from_dict(yaml.safe_load(f))

        self._split_iterators: dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]] = self._build_split_iterators(
            allowed_keys=allowed_keys,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def _build_split_iterators(
        self,
        *,
        allowed_keys: set[str] | None,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance] | None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance] | None,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        from atria_core.datasets._storage._storage_manager import StorageManager

        storage_manager = StorageManager.create(
            self.storage_type,
            data_dir=str(self._path.parent),
            storage_dir=str(self._path.parent),
            config_name=self._path.name,
            num_processes=1,
        )
        result: dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]] = {}
        for split in DatasetSplitType:
            if not storage_manager.split_exists(split):
                continue
            output_transform = (
                train_transform
                if split == DatasetSplitType.train
                else (eval_transform or train_transform)
            )
            result[split] = storage_manager.read_split(
                split=split,
                data_model=self.data_model,
                output_transform=output_transform,  # type: ignore[arg-type]
                allowed_keys=allowed_keys,
            )
        return result

    @property
    def data_dir(self) -> Path:
        return self._path

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

    @property
    def config(self) -> DatasetConfig:
        config_path = self._path / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
        with open(config_path) as f:
            return ModuleConfig.from_dict(yaml.safe_load(f))  # type: ignore[return-value]

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        return self._data_model_cls

    @property
    def metadata(self) -> DatasetMetadata | None:
        return self._metadata_data

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        return self._split_iterators

    def apply_transforms(
        self,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> None:
        for key, split_iterator in self._split_iterators.items():
            if key == DatasetSplitType.train and train_transform is not None:
                split_iterator.output_transform = train_transform
            elif (
                key in {DatasetSplitType.validation, DatasetSplitType.test}
                and eval_transform is not None
            ):
                split_iterator.output_transform = eval_transform

    def split_exists(self, split: DatasetSplitType) -> bool:
        return split in self._split_iterators

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.train not in self._split_iterators:
            raise SplitNotFoundError("Training split iterator is not available.")
        return self._split_iterators[DatasetSplitType.train]

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self._split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.validation not in self._split_iterators:
            raise SplitNotFoundError("Validation split iterator is not available.")
        return self._split_iterators[DatasetSplitType.validation]

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self._split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.test not in self._split_iterators:
            raise SplitNotFoundError("Test split iterator is not available.")
        return self._split_iterators[DatasetSplitType.test]

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self._split_iterators[DatasetSplitType.test] = value
