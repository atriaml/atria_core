from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import tqdm

from atria_core.datasets._storage._msgpack._msgpack_shard_writer import (
    ShardWriterWorker,
)
from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo

logger = get_logger(__name__)

MAX_FAILED_SAMPLES = 10
"""Failed samples tolerated before a parallel split write is aborted."""

WORKER_JOIN_TIMEOUT_SECONDS = 60.0
"""How long to wait for a worker to exit after its sentinel, before treating
it as dead and reporting its shard as unclosed."""

#: One entry per written sample -- (index, "ok", None) on success, or
#: (index, "error", the exception's repr) on failure. Put on a worker's
#: outcome queue as it writes.
_WriteOutcome = tuple[int, Literal["ok", "error"], str | None]

#: A worker's final report, put on the result queue once, right before it
#: exits: the shard it wrote (None if it never got to write anything).
_ShardResult = DatasetShardInfo | None


def _run_worker(
    shard_index: int,
    work_queue: mp.Queue[int | None],
    outcome_queue: mp.Queue[_WriteOutcome],
    result_queue: mp.Queue[_ShardResult],
    dataset: Sequence[Any],
    transform: Callable[[Any], Any] | None,
    split_dir: Path,
) -> None:
    """Write this worker's share of samples, then close its writer and report.

    Runs entirely in one child process: the writer it opens is never shared
    with any other process, so there is no question of who is responsible
    for closing it -- this worker owns it start to finish. Pulls indices off
    `work_queue` until it receives the sentinel `None`, at which point every
    index meant for this worker has already been placed on the queue (FIFO
    ordering means a worker's own sentinel is only consumed after every real
    index queued ahead of it), so closing here is always safe -- there is no
    remaining work this worker could still be assigned afterward.

    Both `dataset[idx]` and `transform` run here, inside the worker -- never
    in the parent process. Indexing a lazy dataset can already do real work
    (decoding images, reading auxiliary files), so doing it in the parent
    before handing items to workers would serialize exactly the work this
    pool exists to parallelize.
    """
    shard_file_pattern = str(Path(split_dir) / f"{shard_index:06d}-%06d.msgpack")
    writer = ShardWriterWorker(
        storage_file_pattern=shard_file_pattern,
        max_shard_size=100_000,
        transform=transform,
    ).load()

    while True:
        idx = work_queue.get()
        if idx is None:
            break
        try:
            raw_item = dataset[idx]
            writer.write(idx, raw_item)
            outcome_queue.put((idx, "ok", None))
        except Exception as e:
            logger.exception(f"Worker failed on sample {idx}")
            outcome_queue.put((idx, "error", repr(e)))

    info = writer.close()
    result_queue.put(info[0] if info else None)


class MultiprocessingParallelSplitWriter:
    """Writes a split to msgpack shards across worker processes.

    Each worker is a long-lived process that owns one persistent shard
    writer for its whole lifetime, reading samples to write off a shared
    queue until it is told there are no more -- so closing a worker's
    writer is always done by that same worker, never delegated to a
    scheduler that might skip it."""

    def __init__(self, num_workers: int = 4) -> None:
        self.num_workers = num_workers

    def write_split(
        self,
        dataset: Sequence[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
    ) -> list[DatasetShardInfo]:
        """Write every sample of a split across worker processes.

        Args:
            dataset: Samples to write. Must be indexable (`Sequence`) --
                only ids are queued here; each worker calls `dataset[idx]`
                itself, since indexing a lazy dataset can already do real
                work. A stream-only source (no `__getitem__`/`len`) can't be
                split into a work queue this way; use a Ray-based writer for
                that instead.
            transform: Applied to each sample before writing, if given.
            split_dir: Directory the shards are written into.

        Returns:
            Metadata for each shard written.

        Raises:
            RuntimeError: If more than MAX_FAILED_SAMPLES samples fail, or if
                a worker never reports back after being sent its sentinel.
        """
        assert isinstance(dataset, Sequence), (
            "MultiprocessingParallelSplitWriter requires an indexable "
            f"(Sequence) dataset; got {type(dataset).__name__}. Use "
            "RayParallelSplitWriter for a stream-only source."
        )

        split_name = split_dir.name
        logger.info(
            f"Writing split {split_name} with {self.num_workers} worker processes..."
        )

        work_queue: mp.Queue[int | None] = mp.Queue()
        outcome_queue: mp.Queue[_WriteOutcome] = mp.Queue()
        result_queue: mp.Queue[_ShardResult] = mp.Queue()

        workers = [
            mp.Process(
                target=_run_worker,
                args=(
                    i,
                    work_queue,
                    outcome_queue,
                    result_queue,
                    dataset,
                    transform,
                    split_dir,
                ),
            )
            for i in range(self.num_workers)
        ]
        for worker in workers:
            worker.start()

        total = len(dataset)
        for idx in range(total):
            work_queue.put(idx)
        for _ in workers:
            work_queue.put(None)

        errors: list[tuple[int, str]] = []
        for _ in tqdm.tqdm(range(total), desc=f"Writing split {split_name}"):
            idx, outcome, error = outcome_queue.get()
            if outcome == "error":
                assert error is not None
                errors.append((idx, error))
                if len(errors) > MAX_FAILED_SAMPLES:
                    for worker in workers:
                        worker.terminate()
                    raise RuntimeError(
                        f"Too many failed samples ({len(errors)}). "
                        f"First failures: {errors[:MAX_FAILED_SAMPLES]}"
                    )

        write_info = self._collect_results(workers, result_queue)

        return [
            replace(shard, shard=i + 1)
            for i, shard in enumerate(write_info)
            if shard is not None
        ]

    def _collect_results(
        self, workers: list[mp.Process], result_queue: mp.Queue[_ShardResult]
    ) -> list[_ShardResult]:
        """Wait for every worker to report its shard, after its sentinel.

        Each worker puts exactly one result right before exiting, so joining
        every worker first guarantees every result is already on the queue
        (or the worker died without reporting one, in which case its join
        still returns -- a dead process doesn't block forever -- and that
        worker is simply missing from `results`).
        """
        results: list[_ShardResult] = []
        dead: list[int] = []
        for worker in workers:
            worker.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
            if worker.is_alive():
                dead.append(worker.pid or -1)
                worker.terminate()
                worker.join()
            else:
                results.append(result_queue.get())

        if dead:
            raise RuntimeError(
                f"{len(dead)} worker(s) never finished within "
                f"{WORKER_JOIN_TIMEOUT_SECONDS}s and were terminated "
                f"(pids: {dead}); their shards may be missing or incomplete."
            )

        return results


class SingleSplitWriter:
    """Writes a split to msgpack shards in the current process."""

    def __init__(self, max_shard_size: int = 100_000) -> None:
        self.max_shard_size = max_shard_size

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
    ) -> list[DatasetShardInfo]:
        """Write every sample of a split in the current process.

        Args:
            dataset: Samples to write.
            transform: Applied to each sample before writing, if given.
            split_dir: Directory the shards are written into.

        Returns:
            Metadata for each shard written.
        """
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
