from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import tqdm

from atria_core.datasets._storage._msgpack._msgpack_shard_writer import (
    ShardWriterWorker,
)
from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo

logger = get_logger(__name__)

_mp_worker_state: tuple[int, ShardWriterWorker] | None = None


def _init_mp_worker(
    index_queue: mp.Queue[int],
    transform: Callable[[Any], Any] | None,
    split_dir: Path,
) -> None:
    global _mp_worker_state

    shard_index = index_queue.get()

    shard_file_pattern = str(Path(split_dir) / f"{shard_index:06d}-%06d.msgpack")

    writer = ShardWriterWorker(
        storage_file_pattern=shard_file_pattern,
        max_shard_size=100_000,
        transform=transform,
    ).load()

    _mp_worker_state = (shard_index, writer)


def _write_shard(sample: tuple[int, Any]) -> tuple[int, str | None]:
    assert _mp_worker_state is not None

    _, writer = _mp_worker_state
    idx, raw_item = sample

    try:
        writer.write(idx, raw_item)
        return idx, None
    except Exception as e:
        logger.exception(f"Worker failed on sample {idx}")
        return idx, repr(e)


def _finish_worker(_: int) -> DatasetShardInfo | None:
    assert _mp_worker_state is not None

    _, writer = _mp_worker_state
    info = writer.close()

    return info[0] if info else None


class MultiprocessingParallelSplitWriter:
    """Ray-free alternative to RayParallelSplitWriter: each of `num_workers`
    Pool workers claims a persistent shard index off `index_queue` once, at
    init time, then individual samples (not chunks) are load-balanced
    across workers via `imap_unordered`."""

    def __init__(self, num_workers: int = 4) -> None:
        self.num_workers = num_workers

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
    ) -> list[DatasetShardInfo]:
        split_name = split_dir.name
        logger.info(
            f"Writing split {split_name} with {self.num_workers} multiprocessing workers..."
        )

        index_queue: mp.Queue[int] = mp.Queue()
        for i in range(self.num_workers):
            index_queue.put(i)

        total = len(dataset) if isinstance(dataset, Sequence) else None
        errors: list[tuple[int, str]] = []

        with mp.Pool(
            processes=self.num_workers,
            initializer=_init_mp_worker,
            initargs=(index_queue, transform, split_dir),
        ) as pool:
            for idx, error in tqdm.tqdm(
                pool.imap_unordered(_write_shard, enumerate(dataset)),
                total=total,
                desc=f"Writing split {split_name}",
            ):
                if error is not None:
                    errors.append((idx, error))

                    if len(errors) > 10:
                        pool.terminate()
                        raise RuntimeError(
                            f"Too many failed samples ({len(errors)}). "
                            f"First failures: {errors[:10]}"
                        )

            write_info = pool.map(_finish_worker, range(self.num_workers))

        return [
            replace(shard, shard=i + 1)
            for i, shard in enumerate(write_info)
            if shard is not None
        ]


class SingleSplitWriter:
    def __init__(self, max_shard_size: int = 100_000) -> None:
        self.max_shard_size = max_shard_size

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
    ) -> list[DatasetShardInfo]:
        split_name = split_dir.name

        writer = ShardWriterWorker(
            storage_file_pattern=str(split_dir / "000000-%06d.msgpack"),
            max_shard_size=self.max_shard_size,
            transform=transform,
        ).load()

        for idx, raw_item in tqdm.tqdm(
            enumerate(dataset), desc=f"Writing split {split_name}"
        ):
            writer.write(idx, raw_item)

        write_info = writer.close()
        return self._finalize_shards(write_info)

    def _finalize_shards(
        self, write_info: list[DatasetShardInfo]
    ) -> list[DatasetShardInfo]:
        write_info = [x for x in write_info if x.nsamples > 0]
        return [replace(shard, shard=i + 1) for i, shard in enumerate(write_info)]
