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
from atria_core.datasets._dataset_builders import ComposedTransform, PreprocessTransform
from atria_core.datasets._exceptions import SplitNotFoundError
from atria_core.datasets._split_iterators import SplitIterator
from atria_core.logger import get_logger
from atria_core.registry import ModuleConfig
from atria_core.types import BaseDataInstance, DatasetMetadata, DatasetSplitType
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
    """Immutable, file-backed dataset produced by Dataset.cache(). Construction
    is lightweight -- call load() to perform all I/O and populate state."""

    __repr_fields__ = {"data_model", "data_dir", "split_iterators", "metadata"}

    def __init__(
        self,
        path: Path | str,
        allowed_keys: set[str] | None = None,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> None:
        self._path = Path(path)
        self._allowed_keys = allowed_keys
        self._train_transform = train_transform
        self._eval_transform = eval_transform
        self._snapshot_data: dict[str, Any] | None = None
        self._metadata_data: DatasetMetadata | None = None
        self._data_model_cls: type[T_BaseDataInstance] | None = None
        self._split_iterators: (
            dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]] | None
        ) = None

    def load(self) -> CachedDataset[T_BaseDataInstance]:
        snapshot_file = self._path / _DEFAULT_SNAPSHOT_PATH
        with open(snapshot_file) as f:
            self._snapshot_data = yaml.load(f, Loader=_SafeTupleLoader)
        assert self._snapshot_data is not None, (
            f"Snapshot file is empty: {snapshot_file}"
        )

        fqn = self._snapshot_data["data_model"]
        module_name, class_name = fqn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self._data_model_cls = getattr(module, class_name)

        metadata_path = self._path / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata_data = DatasetMetadata.from_dict(yaml.safe_load(f))

        self._split_iterators = self._build_split_iterators()
        return self

    @classmethod
    def validate_cache(cls, path: Path | str) -> bool:
        path = Path(path)
        snapshot_file = path / _DEFAULT_SNAPSHOT_PATH
        if not snapshot_file.exists():
            return False
        with open(snapshot_file) as f:
            snapshot = yaml.load(f, Loader=_SafeTupleLoader)
        required_keys = {
            "dataset_class_name",
            "data_model",
            "storage_type",
            "config_name",
            "config_hash",
        }
        return not (required_keys - snapshot.keys())

    @staticmethod
    def _write_yaml(file_path: Path, data: dict[str, Any]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    @classmethod
    def save_dataset_info(
        cls,
        storage_dir: str,
        config_name: str,
        config: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        config_file_path = (
            Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
        )
        logger.info("Saving dataset configuration to %s", config_file_path)
        cls._write_yaml(config_file_path, config)

        metadata_file_path = (
            Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
        )
        logger.info("Saving dataset metadata to %s", metadata_file_path)
        cls._write_yaml(metadata_file_path, metadata)

    @classmethod
    def save_snapshot(
        cls,
        storage_dir: Path | str,
        config_name: str,
        config_hash: str,
        data_model: type[BaseDataInstance],
        storage_type: FileStorageType,
        dataset_name: str | None,
        dataset_class_name: str,
    ) -> None:
        snapshot = {
            "storage_type": storage_type.value,
            "data_model": f"{data_model.__module__}.{data_model.__qualname__}",
            "dataset_name": dataset_name,
            "dataset_class_name": dataset_class_name,
            "config_name": config_name,
            "config_hash": config_hash,
        }
        snapshot_path = Path(storage_dir) / config_name / _DEFAULT_SNAPSHOT_PATH
        logger.info("Saving dataset snapshot to %s", snapshot_path)
        cls._write_yaml(snapshot_path, snapshot)

    def _require_loaded(self) -> None:
        if self._snapshot_data is None:
            raise RuntimeError("CachedDataset state not loaded. Call .load() first.")

    @property
    def data_dir(self) -> Path:
        return self._path

    @property
    def dataset_name(self) -> str | None:
        self._require_loaded()
        assert self._snapshot_data is not None
        return self._snapshot_data.get("dataset_name")

    @property
    def dataset_class_name(self) -> str:
        self._require_loaded()
        assert self._snapshot_data is not None
        return str(self._snapshot_data["dataset_class_name"])

    @property
    def storage_type(self) -> FileStorageType:
        self._require_loaded()
        assert self._snapshot_data is not None
        return FileStorageType(self._snapshot_data["storage_type"])

    @property
    def config_name(self) -> str:
        self._require_loaded()
        assert self._snapshot_data is not None
        return str(self._snapshot_data["config_name"])

    @property
    def config_hash(self) -> str:
        self._require_loaded()
        assert self._snapshot_data is not None
        return str(self._snapshot_data["config_hash"])

    @property
    def config(self) -> DatasetConfig:
        config_path = self._path / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
        with open(config_path) as f:
            return ModuleConfig.from_dict(yaml.safe_load(f))  # type: ignore[return-value]

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        self._require_loaded()
        assert self._data_model_cls is not None
        return self._data_model_cls

    @property
    def metadata(self) -> DatasetMetadata | None:
        self._require_loaded()
        return self._metadata_data

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        self._require_loaded()
        assert self._split_iterators is not None
        return self._split_iterators

    def _build_split_iterators(
        self,
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
            iterator = storage_manager.read_split(
                split=split, data_model=self.data_model, allowed_keys=self._allowed_keys
            )
            tf = (
                self._train_transform
                if split == DatasetSplitType.train
                else (self._eval_transform or self._train_transform)
            )
            iterator.output_transform = (
                ComposedTransform([PreprocessTransform(), tf])
                if tf is not None
                else PreprocessTransform()
            )
            result[split] = iterator
        return result

    def apply_transforms(
        self,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> None:
        assert self._split_iterators is not None, "Split iterators not loaded"
        for key, split_iterator in self._split_iterators.items():
            if key == DatasetSplitType.train and train_transform is not None:
                split_iterator.output_transform = ComposedTransform(  # type: ignore[assignment]
                    [PreprocessTransform(), train_transform]
                )
            elif (
                key in {DatasetSplitType.validation, DatasetSplitType.test}
                and eval_transform is not None
            ):
                split_iterator.output_transform = ComposedTransform(  # type: ignore[assignment]
                    [PreprocessTransform(), eval_transform]
                )

    def split_exists(self, split: DatasetSplitType) -> bool:
        return split in self.split_iterators

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.train not in self.split_iterators:
            raise SplitNotFoundError("Training split iterator is not available.")
        return self.split_iterators[DatasetSplitType.train]

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self.split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.validation not in self.split_iterators:
            raise SplitNotFoundError("Validation split iterator is not available.")
        return self.split_iterators[DatasetSplitType.validation]

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self.split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.test not in self.split_iterators:
            raise SplitNotFoundError("Test split iterator is not available.")
        return self.split_iterators[DatasetSplitType.test]

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self.split_iterators[DatasetSplitType.test] = value
