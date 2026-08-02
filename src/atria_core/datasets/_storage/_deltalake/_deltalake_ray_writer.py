from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import ray
import tqdm

from atria_core.datasets._storage._deltalake._deltalake_row_writer import (
    _DeltaBatchWriter,
    _write_items,
)
from atria_core.datasets._storage._ray_utils import RAY_RUNTIME_ENV
from atria_core.logger import get_logger

logger = get_logger(__name__)


@ray.remote
class DeltalakeShardWriterActor:
    def __init__(
        self,
        artifacts_dir: Path,
        transform: Callable[[Any], Any] | None,
    ) -> None:
        self._artifacts_dir = artifacts_dir
        self._transform = transform

    def write(self, sample_tuple: tuple[int, Any]) -> list[dict[str, Any]]:
        idx, raw_item = sample_tuple
        try:
            return _write_items(raw_item, self._transform, self._artifacts_dir)
        except Exception:
            logger.exception(f"Error writing sample at index {idx}")
            return []


class RayParallelDeltalakeWriter:
    def __init__(
        self,
        num_workers: int = 4,
        max_concurrent_tasks_limit: int = 128,
        max_memory_per_actor: int = 500 * 1024 * 1024,
        max_memory: int = 1_000_000_000,
    ) -> None:
        self.num_workers = num_workers
        self.max_concurrent_tasks_limit = max_concurrent_tasks_limit
        self.max_memory_per_actor = max_memory_per_actor
        self.max_memory = max_memory

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
        artifacts_dir: Path,
    ) -> None:
        split_name = split_dir.name
        logger.info(f"Writing split {split_name} with {self.num_workers} Ray actors...")

        ray.init(
            num_cpus=self.num_workers,
            local_mode=self.num_workers == 1,
            runtime_env=RAY_RUNTIME_ENV,
        )

        actors = [
            DeltalakeShardWriterActor.options(  # type: ignore[attr-defined]
                memory=self.max_memory_per_actor
            ).remote(artifacts_dir=artifacts_dir, transform=transform)
            for _ in range(self.num_workers)
        ]

        batch_writer = _DeltaBatchWriter(split_dir, self.max_memory)

        try:
            pending_tasks = []
            actor_iterator = itertools.cycle(actors)

            for idx, raw_item in tqdm.tqdm(
                enumerate(dataset), desc=f"Writing split {split_name}"
            ):
                actor = next(actor_iterator)
                pending_tasks.append(actor.write.remote((idx, raw_item)))

                if len(pending_tasks) >= self.max_concurrent_tasks_limit:
                    ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)
                    for row_list in ray.get(ready_tasks):
                        batch_writer.add(row_list)

            for row_list in ray.get(pending_tasks):
                batch_writer.add(row_list)
            batch_writer.finish()

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, shutting down Ray actors...")
            raise
        except Exception as e:
            logger.exception("Error during Ray parallel split writing")
            raise e
        finally:
            if ray.is_initialized():
                ray.shutdown()
