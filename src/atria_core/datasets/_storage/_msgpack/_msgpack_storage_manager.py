from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from atria_core.datasets._storage._msgpack._msgpack_local_writer import (
    MultiprocessingParallelSplitWriter,
    SingleSplitWriter,
)
from atria_core.datasets._storage._msgpack._msgpack_ray_writer import (
    RayParallelSplitWriter,
)
from atria_core.datasets._storage._msgpack._msgpack_shard_list_dataset import (
    MsgpackShardListDataset,
)
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo, DatasetSplitType

logger = get_logger(__name__)


class MsgpackStorageManager(StorageManager):
    storage_prefix: ClassVar[str] = "msgpack"

    def __init__(
        self,
        data_dir: str | Path,
        storage_dir: str | Path,
        config_name: str,
        num_processes: int = 8,
        max_shard_size: int = 100_000,
        name_suffix: str = "",
        use_ray: bool = False,
        store_images_to_files: bool = False,
    ) -> None:
        self.max_shard_size = max_shard_size
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
        offsets = list(self.split_dir(split).glob("*.msgpack.offsets"))
        return len(self.split_files(split)) > 0 and len(offsets) > 0

    def split_files(self, split: DatasetSplitType) -> list[Path]:
        return list(self.split_dir(split).glob("*.msgpack"))

    def _write_split_internal(
        self, split: DatasetSplitType, split_iterator: Any
    ) -> None:
        split_dir = self.split_dir(split)
        logger.info(
            f"Writing dataset split {split.value} to {split_dir} "
            f"({'parallel' if self.num_processes > 1 else 'single'} mode)"
        )

        dataset = split_iterator.base_iterator
        transform = split_iterator.transform

        writer: (
            RayParallelSplitWriter
            | MultiprocessingParallelSplitWriter
            | SingleSplitWriter
        )
        if self.num_processes > 1 and self.use_ray:
            writer = RayParallelSplitWriter(
                num_workers=self.num_processes, max_shard_size=self.max_shard_size
            )
        elif self.num_processes > 1:
            writer = MultiprocessingParallelSplitWriter(num_workers=self.num_processes)
        else:
            writer = SingleSplitWriter(max_shard_size=self.max_shard_size)

        write_info = writer.write_split(dataset, transform, split_dir)
        self._log_write_results(write_info, split)

    def _log_write_results(
        self, write_info: list[DatasetShardInfo], split: DatasetSplitType
    ) -> None:
        total_samples = sum(shard.nsamples for shard in write_info)
        logger.info(
            f"Successfully wrote {total_samples} samples to {len(write_info)} shards "
            f"for split {split.value}"
        )

    def read_split(self, split: DatasetSplitType) -> Sequence[Any]:
        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )
        return MsgpackShardListDataset(self.split_files(split))
