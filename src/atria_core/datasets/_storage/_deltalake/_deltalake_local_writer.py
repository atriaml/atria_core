from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import tqdm

from atria_core.datasets._storage._deltalake._deltalake_row_writer import (
    _DeltaBatchWriter,
    _write_items,
)
from atria_core.logger import get_logger

logger = get_logger(__name__)

_mp_delta_worker_state: tuple[Path, Callable[[Any], Any] | None] | None = None


def _init_mp_delta_worker(
    artifacts_dir: Path, transform: Callable[[Any], Any] | None
) -> None:
    global _mp_delta_worker_state
    _mp_delta_worker_state = (artifacts_dir, transform)


def _convert_row(sample: tuple[int, Any]) -> list[dict[str, Any]]:
    """Runs in a worker process, dispatched via Pool.imap_unordered: converts
    one raw item to its flattened row(s) (applying `transform`, hoisting
    bytes to artifacts). The actual Delta Lake writes still happen in the
    main process -- Delta appends must be ordered, unlike msgpack's
    independent per-worker shard files, so workers never write directly."""
    assert _mp_delta_worker_state is not None, "Worker was not initialized"
    artifacts_dir, transform = _mp_delta_worker_state
    idx, raw_item = sample
    try:
        return _write_items(raw_item, transform, artifacts_dir)
    except Exception:
        logger.exception(f"Error writing sample at index {idx}")
        return []


class MultiprocessingParallelDeltalakeWriter:
    """Writes a split to Delta Lake using multiprocessing workers for
    row conversion.

    Workers only convert raw items into rows, one sample at a time. The Delta
    Lake writes themselves happen in this process, in order and in batches, so
    row ordering is preserved."""

    def __init__(self, num_workers: int = 4, max_memory: int = 1_000_000_000) -> None:
        self.num_workers = num_workers
        self.max_memory = max_memory

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
        artifacts_dir: Path,
    ) -> None:
        """Write every sample of a split to a Delta table.

        Args:
            dataset: Samples to write.
            transform: Applied to each sample before writing, if given.
            split_dir: Directory the Delta table is written into.
            artifacts_dir: Directory large binary fields are hoisted into.
        """
        split_name = split_dir.name
        logger.info(
            f"Writing split {split_name} with {self.num_workers} multiprocessing workers..."
        )

        batch_writer = _DeltaBatchWriter(split_dir, self.max_memory)
        total = len(dataset) if isinstance(dataset, Sequence) else None

        with mp.Pool(
            processes=self.num_workers,
            initializer=_init_mp_delta_worker,
            initargs=(artifacts_dir, transform),
        ) as pool:
            for rows in tqdm.tqdm(
                pool.imap_unordered(_convert_row, enumerate(dataset)),
                total=total,
                desc=f"Writing split {split_name}",
            ):
                batch_writer.add(rows)

        batch_writer.finish()


class SingleDeltalakeWriter:
    """Writes a split to Delta Lake in the current process."""

    def __init__(self, max_memory: int = 1_000_000_000) -> None:
        self.max_memory = max_memory

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
        artifacts_dir: Path,
    ) -> None:
        """Write every sample of a split to a Delta table.

        Args:
            dataset: Samples to write.
            transform: Applied to each sample before writing, if given.
            split_dir: Directory the Delta table is written into.
            artifacts_dir: Directory large binary fields are hoisted into.
        """
        split_name = split_dir.name
        batch_writer = _DeltaBatchWriter(split_dir, self.max_memory)

        try:
            for idx, raw_item in tqdm.tqdm(
                enumerate(dataset), desc=f"Writing split {split_name}"
            ):
                try:
                    batch_writer.add(_write_items(raw_item, transform, artifacts_dir))
                except Exception:
                    logger.exception(f"Error writing sample at index {idx}")
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, stopping split writing...")
            raise

        batch_writer.finish()
