from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from atria_core.datasets._common import FileStorageType
from atria_core.datasets._split_iterators import SplitIterator
from atria_core.logger import get_logger
from atria_core.types import BaseDataInstance, DatasetSplitType

logger = get_logger(__name__)


class StorageManager(ABC):
    """Base class for on-disk dataset storage backends: writes/reads splits
    at a given storage_dir/config_name. Deciding *what* that path should be
    (cache uniqueness -- dataset config hash, transform hash, storage
    backend) is Cacher's job, not this class's; StorageManager only knows
    how to store what it's told, where it's told.
    """

    storage_prefix: ClassVar[str]

    def __init__(
        self,
        data_dir: str | Path,
        storage_dir: str | Path,
        config_name: str,
        num_processes: int = 8,
        name_suffix: str = "",
        use_ray: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.storage_dir = Path(storage_dir)
        self.config_name = config_name
        self.num_processes = num_processes
        self.name_suffix = name_suffix
        self.use_ray = use_ray

        self._setup_directories()

    @classmethod
    def resolve_class(cls, cached_storage_type: FileStorageType) -> type[StorageManager]:
        if cached_storage_type == FileStorageType.DELTALAKE:
            from atria_core.datasets._storage._deltalake_storage_manager import (
                DeltalakeStorageManager,
            )

            return DeltalakeStorageManager
        elif cached_storage_type == FileStorageType.MSGPACK:
            from atria_core.datasets._storage._msgpack_storage_manager import (
                MsgpackStorageManager,
            )

            return MsgpackStorageManager
        raise ValueError(f"Unsupported storage type: {cached_storage_type}")

    @classmethod
    def create(
        cls,
        cached_storage_type: FileStorageType,
        data_dir: str | Path,
        num_processes: int = 8,
        name_suffix: str = "",
        *,
        storage_dir: str | Path,
        config_name: str,
        use_ray: bool = False,
    ) -> StorageManager:
        """Resolve the concrete StorageManager for `cached_storage_type` and
        instantiate it at the given, already-computed `storage_dir`/
        `config_name`. use_ray=False (default) parallelizes writes with
        plain multiprocessing, which has lower overhead for the common
        case; use_ray=True opts into Ray actors instead."""
        storage_manager_cls = cls.resolve_class(cached_storage_type)
        return storage_manager_cls(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
            use_ray=use_ray,
        )

    def _setup_directories(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        assert self.storage_dir.is_dir(), (
            f"Storage directory {self.storage_dir} must be a directory."
        )
        (self.storage_dir / self.config_name).mkdir(parents=True, exist_ok=True)

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
