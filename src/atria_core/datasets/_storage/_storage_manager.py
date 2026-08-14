from __future__ import annotations

import hashlib
import io
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

from atria_core.datasets._common import FileStorageType
from atria_core.logger import get_logger
from atria_core.types import (
    DataInstance,
    DatasetSplitType,
    Image,
    ImageInstance,
    SinglePageDocumentInstance,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class _StoreImagesToFiles:
    artifacts_dir: Path

    def __call__(self, sample: DataInstance) -> DataInstance:
        if isinstance(sample, ImageInstance):
            return replace(sample, image=self._store(sample.image))
        if isinstance(sample, SinglePageDocumentInstance) and isinstance(
            sample.visual, Image
        ):
            return replace(sample, visual=self._store(sample.visual))
        return sample

    def _store(self, image: Image) -> Image:
        if image.file_path is not None:
            return replace(image, content=None)

        buffer = io.BytesIO()
        image.require_content().save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()
        image_path = self.artifacts_dir / digest[:2] / f"{digest}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with image_path.open("xb") as output:
                output.write(image_bytes)
        except FileExistsError:
            pass
        return replace(image, file_path=str(image_path), content=None)


class StorageManager(ABC):
    """Base class for on-disk dataset storage backends.

    Writes and reads splits under a given `storage_dir`/`config_name`. It
    stores what it is told, where it is told; choosing a location that keeps
    distinct caches apart is the caller's responsibility. Split identity is
    never inferred, so `split` is an explicit parameter on every operation.
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
        store_images_to_files: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.storage_dir = Path(storage_dir)
        self.config_name = config_name
        self.num_processes = num_processes
        self.name_suffix = name_suffix
        self.use_ray = use_ray
        self.store_images_to_files = store_images_to_files

        self._setup_directories()

    @classmethod
    def resolve_class(
        cls, cached_storage_type: FileStorageType
    ) -> type[StorageManager]:
        """Return the StorageManager subclass handling `cached_storage_type`.

        Raises:
            ValueError: If the storage type has no registered backend.
        """
        if cached_storage_type == FileStorageType.DELTALAKE:
            from atria_core.datasets._storage._deltalake._deltalake_storage_manager import (
                DeltalakeStorageManager,
            )

            return DeltalakeStorageManager
        elif cached_storage_type == FileStorageType.MSGPACK:
            from atria_core.datasets._storage._msgpack._msgpack_storage_manager import (
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
        store_images_to_files: bool = False,
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
            store_images_to_files=store_images_to_files,
        )

    def _setup_directories(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        assert self.storage_dir.is_dir(), (
            f"Storage directory {self.storage_dir} must be a directory."
        )
        (self.storage_dir / self.config_name).mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: DatasetSplitType) -> Path:
        """Return the directory holding `split`, creating it if needed."""
        split_dir = self.storage_dir / self.config_name / split.value / self.name_suffix
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def dataset_exists(self) -> bool:
        """Whether any split has been written to this storage location."""
        return bool(self.get_splits())

    def get_splits(self) -> list[DatasetSplitType]:
        """Return every split present in this storage location."""
        return [split for split in DatasetSplitType if self.split_exists(split)]

    def purge_split(self, split: DatasetSplitType) -> None:
        """Delete `split` and everything written for it."""
        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(f"Purging dataset split {split.value} from storage {split_dir}")
            shutil.rmtree(split_dir)

    def write_split(self, split: DatasetSplitType, split_iterator: Any) -> None:
        """Write every sample of `split` to storage, purging it on failure.

        Args:
            split: Which split is being written.
            split_iterator: Samples to write.

        Raises:
            Exception: Re-raised from the backend after the partial split is
                purged, so a failed write never leaves a half-written split.
        """
        try:
            if self.store_images_to_files:
                artifacts_dir = (
                    self.storage_dir / self.config_name / "artifacts" / "images"
                )
                split_iterator = split_iterator.with_transform(
                    _StoreImagesToFiles(artifacts_dir)
                )
            self._write_split_internal(split, split_iterator)
        except (Exception, KeyboardInterrupt) as e:
            self.purge_split(split)
            error_msg = (
                "KeyboardInterrupt detected. Stopping dataset preparation..."
                if isinstance(e, KeyboardInterrupt)
                else f"Error while writing dataset split {split.value} to storage. Cleaning up..."
            )
            raise type(e)(error_msg) from e

    @abstractmethod
    def split_exists(self, split: DatasetSplitType) -> bool:
        """Whether `split` has already been written here."""
        raise NotImplementedError

    @abstractmethod
    def _write_split_internal(
        self, split: DatasetSplitType, split_iterator: Any
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_split(self, split: DatasetSplitType) -> Sequence[Any] | Iterable[Any]:
        """Returns the *raw*, undecoded split contents (plain dicts) --
        decoding into data_model instances happens through Dataset's own
        generic input-transform wrapping, not here."""
        raise NotImplementedError
