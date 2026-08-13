from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from atria_core.datasets._storage._deltalake._deltalake_local_writer import (
    MultiprocessingParallelDeltalakeWriter,
    SingleDeltalakeWriter,
)
from atria_core.datasets._storage._deltalake._deltalake_ray_writer import (
    RayParallelDeltalakeWriter,
)
from atria_core.datasets._storage._deltalake._deltalake_reader import (
    InMemoryDeltalakeReader,
    LocalDeltalakeReader,
)
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType

logger = get_logger(__name__)


class DeltalakeStorageManager(StorageManager):
    storage_prefix: ClassVar[str] = "delta"

    def __init__(
        self,
        data_dir: str | Path,
        storage_dir: str | Path,
        config_name: str,
        num_processes: int = 8,
        max_memory: int = 1_000_000_00,
        name_suffix: str = "",
        use_ray: bool = False,
        store_images_to_files: bool = False,
    ) -> None:
        self.max_memory = max_memory
        super().__init__(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
            use_ray=use_ray,
            store_images_to_files=store_images_to_files,
        )

    def split_exists(self, split: DatasetSplitType) -> bool:
        return (self.split_dir(split) / "_delta_log").exists()

    def _write_split_internal(
        self, split: DatasetSplitType, split_iterator: Any
    ) -> None:
        split_dir = self.split_dir(split)
        write_dir = self.storage_dir / self.config_name
        artifacts_dir = write_dir / "data" / split.value / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Writing dataset split {split.value} to {split_dir} "
            f"({'parallel' if self.num_processes > 1 else 'single'} mode)"
        )

        dataset = split_iterator.base_iterator
        transform = split_iterator.transform

        writer: (
            RayParallelDeltalakeWriter
            | MultiprocessingParallelDeltalakeWriter
            | SingleDeltalakeWriter
        )
        if self.num_processes > 1 and self.use_ray:
            writer = RayParallelDeltalakeWriter(
                num_workers=self.num_processes, max_memory=self.max_memory
            )
        elif self.num_processes > 1:
            writer = MultiprocessingParallelDeltalakeWriter(
                num_workers=self.num_processes, max_memory=self.max_memory
            )
        else:
            writer = SingleDeltalakeWriter(max_memory=self.max_memory)

        writer.write_split(dataset, transform, split_dir, artifacts_dir)

    def read_split(
        self, split: DatasetSplitType, streaming_mode: bool = False
    ) -> Sequence[Any]:
        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        reader_cls = LocalDeltalakeReader if streaming_mode else InMemoryDeltalakeReader
        return reader_cls(
            table_path=str(self.split_dir(split=split)),
            storage_dir=str(self.storage_dir),
            config_name=self.config_name,
        )
