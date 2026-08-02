from __future__ import annotations

import hashlib
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
from atria_core.datasets._dataset_builders import _default_data_dir, _validate_data_dir
from atria_core.datasets._split_iterators import SplitIterator
from atria_core.logger import get_logger
from atria_core.types import BaseDataInstance, DatasetSplitType

if TYPE_CHECKING:
    from atria_core.datasets._dataset import Dataset

logger = get_logger(__name__)


def _get_combined_transform_hash(
    preprocess_train_transform: Any | None, preprocess_eval_transform: Any | None
) -> str:
    """`atria_core.transforms.DataTransform` (with its `.hash` property)
    hasn't been ported yet, so `preprocess_*_transform` are loosely typed
    `Any` here rather than importing a class that doesn't exist."""
    t = (
        preprocess_train_transform.hash
        if preprocess_train_transform is not None
        else "none"
    )
    e = (
        preprocess_eval_transform.hash
        if preprocess_eval_transform is not None
        else "none"
    )
    return hashlib.md5(f"train:{t}|eval:{e}".encode()).hexdigest()[:8]


class StorageManager(ABC):
    """Base class for on-disk dataset storage backends.

    Each concrete backend owns a `storage_prefix` that is folded into the
    top-level unique cache path returned by `compute_cache_path`, which is
    the single source of truth for cache uniqueness. This guarantees that
    switching storage backend for the same dataset config never reuses, or
    collides with, another backend's cache directory.
    """

    storage_prefix: ClassVar[str]

    def __init__(
        self,
        data_dir: str | Path,
        storage_dir: str | Path,
        config_name: str,
        num_processes: int = 8,
        name_suffix: str = "",
    ) -> None:
        self.data_dir = data_dir
        self.storage_dir = Path(storage_dir)
        self.config_name = config_name
        self.num_processes = num_processes
        self.name_suffix = name_suffix

        self._setup_directories()

    @classmethod
    def create(
        cls,
        cached_storage_type: FileStorageType,
        data_dir: str | Path,
        num_processes: int = 8,
        name_suffix: str = "",
        *,
        storage_dir: str | Path | None = None,
        config_name: str | None = None,
        dataset: Dataset[Any, Any] | None = None,
        preprocess_train_transform: Any | None = None,
        preprocess_eval_transform: Any | None = None,
    ) -> StorageManager:
        """Resolve the concrete StorageManager for `cached_storage_type` and instantiate it.

        Pass `dataset` (optionally with preprocess transforms) to have the
        unique cache path computed automatically via `compute_cache_path`;
        otherwise pass `storage_dir`/`config_name` directly for an
        already-known cache location (e.g. reloading an existing snapshot).
        """
        if cached_storage_type == FileStorageType.DELTALAKE:
            from atria_core.datasets._storage._deltalake_storage_manager import (
                DeltalakeStorageManager,
            )

            storage_manager_cls: type[StorageManager] = DeltalakeStorageManager
        elif cached_storage_type == FileStorageType.MSGPACK:
            from atria_core.datasets._storage._msgpack_storage_manager import (
                MsgpackStorageManager,
            )

            storage_manager_cls = MsgpackStorageManager
        else:
            raise ValueError(f"Unsupported storage type: {cached_storage_type}")

        if dataset is not None:
            unique_path = storage_manager_cls.compute_cache_path(
                dataset, data_dir, preprocess_train_transform, preprocess_eval_transform
            )
            storage_dir, config_name = str(unique_path.parent), unique_path.name

        assert storage_dir is not None and config_name is not None, (
            "create() requires either `dataset` or explicit `storage_dir`/`config_name`."
        )
        return storage_manager_cls(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
        )

    def _setup_directories(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        assert self.storage_dir.is_dir(), (
            f"Storage directory {self.storage_dir} must be a directory."
        )
        (self.storage_dir / self.config_name).mkdir(parents=True, exist_ok=True)

    @classmethod
    def compute_cache_path(
        cls,
        dataset: Dataset[Any, Any],
        data_dir: str | Path | None,
        preprocess_train_transform: Any | None = None,
        preprocess_eval_transform: Any | None = None,
    ) -> Path:
        """Single source of truth for cache uniqueness.

        Folds the dataset config name/hash, this backend's storage_prefix,
        and (if given) a combined preprocess-transform hash into the
        top-level cache directory name, so caches for different storage
        backends, or different preprocess transforms, never collide.
        """
        resolved = _validate_data_dir(data_dir or _default_data_dir(dataset))
        storage_dir = Path(resolved) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
        config_name = (
            f"{cls.storage_prefix}/{dataset.config.config_name}-{dataset.config.hash}"
        )
        if (
            preprocess_train_transform is not None
            or preprocess_eval_transform is not None
        ):
            config_name += "-" + _get_combined_transform_hash(
                preprocess_train_transform, preprocess_eval_transform
            )
        return storage_dir / config_name

    def split_dir(self, split: DatasetSplitType) -> Path:
        split_dir = self.storage_dir / self.config_name / split.value / self.name_suffix
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def dataset_exists(self) -> bool:
        return bool(self.get_splits())

    def get_splits(self) -> list[DatasetSplitType]:
        return [split for split in DatasetSplitType if self.split_exists(split)]

    def purge_split(self, split: DatasetSplitType) -> None:
        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(f"Purging dataset split {split.value} from storage {split_dir}")
            shutil.rmtree(split_dir)

    def write_split(self, split_iterator: SplitIterator[Any]) -> None:
        try:
            self._write_split_internal(split_iterator)
        except (Exception, KeyboardInterrupt) as e:
            self.purge_split(split_iterator.split)
            error_msg = (
                "KeyboardInterrupt detected. Stopping dataset preparation..."
                if isinstance(e, KeyboardInterrupt)
                else f"Error while writing dataset split {split_iterator.split.value} to storage. Cleaning up..."
            )
            raise type(e)(error_msg) from e

    @abstractmethod
    def split_exists(self, split: DatasetSplitType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _write_split_internal(self, split_iterator: SplitIterator[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_split(
        self,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
        output_transform: Callable[
            [BaseDataInstance], BaseDataInstance | list[BaseDataInstance]
        ]
        | None = None,
        allowed_keys: set[str] | None = None,
    ) -> SplitIterator[Any]:
        raise NotImplementedError
