from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import ray
import tqdm

from atria_core.datasets._storage._msgpack._msgpack_shard_writer import (
    DuplicateKeyError,
    ShardWriterWorker,
)
from atria_core.datasets._storage._ray_utils import RAY_RUNTIME_ENV
from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo

logger = get_logger(__name__)


@ray.remote
class ShardWriterActor:
    def __init__(
        self,
        worker_id: int,
        split_dir: Path,
        max_shard_size: int,
        transform: Callable[[Any], Any] | None,
    ) -> None:
        shard_file_pattern = str(split_dir / f"{worker_id:06d}-%06d.msgpack")
        self.writer = ShardWriterWorker(
            storage_file_pattern=shard_file_pattern,
            max_shard_size=max_shard_size,
            transform=transform,
        ).load()
        self.error_count = 0

    def write(self, sample_tuple: tuple[int, Any]) -> bool:
        idx, raw_item = sample_tuple
        try:
            self.writer.write(idx, raw_item)
        except DuplicateKeyError:
            logger.error(f"Duplicate key at index {idx}, skipping")
        except Exception:
            logger.exception(f"Error writing sample at index {idx}")
            self.error_count += 1

        if self.error_count >= 10:
            logger.error("Too many errors encountered. Stopping writer.")
            raise RuntimeError("Too many errors in ShardWriterActor")
        return True

    def close(self) -> list[DatasetShardInfo]:
        return self.writer.close()


class RayParallelSplitWriter:
    """Parallel split writer using Ray actors: raw items are round-robin
    dispatched to per-actor msgpack shard writers (each applying
    `transform` itself), with a bounded in-flight task window (`ray.wait`)
    so the driver doesn't outrun the actors."""

    def __init__(
        self,
        num_workers: int = 4,
        max_shard_size: int = 100_000,
        max_concurrent_tasks_limit: int = 128,
        max_memory_per_actor: int = 500 * 1024 * 1024,
    ) -> None:
        self.num_workers = num_workers
        self.max_shard_size = max_shard_size
        self._max_concurrent_tasks_limit = max_concurrent_tasks_limit
        self._max_memory_per_actor = max_memory_per_actor
        self.actors: list[Any] = []

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
    ) -> list[DatasetShardInfo]:
        try:
            split_name = split_dir.name
            logger.info(
                f"Writing split {split_name} with {self.num_workers} Ray actors..."
            )

            ray.init(
                num_cpus=self.num_workers,
                local_mode=self.num_workers == 1,
                runtime_env=RAY_RUNTIME_ENV,
            )

            self.actors = [
                ShardWriterActor.options(  # type: ignore[attr-defined]
                    memory=self._max_memory_per_actor
                ).remote(
                    worker_id=i,
                    split_dir=split_dir,
                    max_shard_size=self.max_shard_size,
                    transform=transform,
                )
                for i in range(self.num_workers)
            ]

            pending_tasks = []
            actor_iterator = itertools.cycle(self.actors)

            for idx, raw_item in tqdm.tqdm(
                enumerate(dataset), desc=f"Writing split {split_name}"
            ):
                actor = next(actor_iterator)
                pending_tasks.append(actor.write.remote((idx, raw_item)))

                if len(pending_tasks) >= self._max_concurrent_tasks_limit:
                    ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)
                    ray.get(ready_tasks)

            ray.get(pending_tasks)

            write_info_per_actor = ray.get(
                [actor.close.remote() for actor in self.actors]
            )
            write_info = [
                x for shard in write_info_per_actor for x in shard if x.nsamples > 0
            ]
            return [replace(shard, shard=i + 1) for i, shard in enumerate(write_info)]
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, shutting down Ray actors...")
            raise
        except Exception as e:
            logger.exception("Error during Ray parallel split writing")
            raise e
        finally:
            if ray.is_initialized():
                ray.shutdown()
