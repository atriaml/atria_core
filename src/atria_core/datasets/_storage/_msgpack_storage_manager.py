from __future__ import annotations

import itertools
import multiprocessing as mp
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Self

import ray
import tqdm
from datadings.writer import Writer

from atria_core.datasets._storage._msgpack_shard_list_dataset import (
    MsgpackShardListDataset,
)
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo, DatasetSplitType

logger = get_logger(__name__)

_RAY_RUNTIME_ENV = {"env_vars": {"PYTHONPATH": ":".join(sys.path)}}


class DuplicateKeyError(Exception):
    pass


class MsgpackFileWriter(Writer):  # type: ignore[misc]
    def _write_data(self, key: str, packed: bytes) -> None:
        if key in self._keys_set:
            raise DuplicateKeyError(f"Duplicate key {key!r} not allowed.")
        self._keys.append(key)
        self._keys_set.add(key)
        self._hash.update(packed)
        self._outfile.write(packed)
        self._offsets.append(self._outfile.tell())
        self.written += 1

    def write(self, sample: dict[str, Any]) -> int:
        assert "key" in sample, "Sample must contain a unique 'key' value."
        self._write(sample["key"], sample)
        return self.written  # type: ignore[no-any-return]


class MsgpackShardWriter:
    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Callable[[str], None] | None = None,
        start_shard: int = 0,
        verbose: int = 0,
        opener: Callable[[str], Any] | None = None,
        **kw: Any,
    ) -> None:
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.writer: MsgpackFileWriter | None = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname: str | None = None
        self.opener = opener
        self.next_stream()

    def next_stream(self) -> None:
        self.finish()
        self.fname = self.pattern % self.shard
        if self.verbose:
            logger.info(
                f"# Writing {self.fname}, {self.count} samples, {self.size / 1e9:.1f} GB, {self.total} total samples."
            )
        self.shard += 1
        if self.opener:
            self.writer = MsgpackFileWriter(
                self.opener(self.fname), **self.kw, disable=True
            )
        else:
            self.writer = MsgpackFileWriter(self.fname, **self.kw, disable=True)
        self.count = 0
        self.size = 0

    def write(self, obj: dict[str, Any]) -> None:
        if (
            self.writer is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        assert self.writer is not None
        size = self.writer.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.writer = None

    def close(self) -> None:
        self.finish()
        del self.writer
        del self.shard
        del self.count
        del self.size

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any, **kw: Any) -> None:
        self.close()


class ShardWriterWorker:
    """Wraps `MsgpackShardWriter`, tracking per-shard `DatasetShardInfo` as
    shards rotate so callers get back sharding metadata, not just files.
    `transform` is a plain Callable[[Any], Any] (or None) applied to each
    raw item on write -- may return a single instance or a list of them."""

    def __init__(
        self,
        storage_file_pattern: str,
        max_shard_size: int,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self._storage_file_pattern = storage_file_pattern
        self._max_shard_size = max_shard_size
        self._transform = transform
        self._writer: MsgpackShardWriter | None = None
        self._write_info: list[DatasetShardInfo] = []

    def load(self) -> Self:
        self._writer = MsgpackShardWriter(
            self._storage_file_pattern, maxcount=self._max_shard_size, overwrite=True
        )
        assert self._writer.fname is not None, "Writer filename is None."
        self._write_info.append(DatasetShardInfo(url=self._writer.fname))
        return self

    def write(self, index: int, raw_item: Any) -> None:
        if self._writer is None:
            raise RuntimeError("ShardWriter is not loaded. Call `load()` first.")
        assert self._writer.fname is not None, "Writer filename is None."

        result = self._transform(raw_item) if self._transform is not None else raw_item
        items = result if isinstance(result, list) else [result]

        if self._writer.shard != self._write_info[-1].shard:
            self._write_info.append(
                DatasetShardInfo(
                    url=self._writer.fname,
                    shard=self._writer.shard,
                    nsamples=self._writer.count,
                    filesize=self._writer.size,
                )
            )

        for item in items:
            self._writer.write({**item.to_dict(), "key": item.key})

        # DatasetShardInfo is frozen, so the running shard-info entry is
        # replaced (not mutated) each write.
        self._write_info[-1] = replace(
            self._write_info[-1],
            url=self._writer.fname,
            shard=self._writer.shard,
            nsamples=self._writer.count,
            filesize=self._writer.size,
        )

    def close(self) -> list[DatasetShardInfo]:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._write_info


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
                runtime_env=_RAY_RUNTIME_ENV,
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


_mp_worker_state: tuple[Callable[[Any], Any] | None, Path] | None = None


def _init_mp_worker(
    index_queue: mp.Queue,
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


def _write_shard(sample: tuple[int, Any]) -> None:
    assert _mp_worker_state is not None

    shard_index, writer = _mp_worker_state

    idx, raw_item = sample
    try:
        writer.write(idx, raw_item)
    except DuplicateKeyError:
        logger.error(f"Duplicate key at index {idx}, skipping")
    except Exception:
        logger.exception(f"Error writing sample at index {idx}")


def _finish_worker(_):
    assert _mp_worker_state is not None

    _, writer = _mp_worker_state
    info = writer.close()

    return info[0] if info else None


class MultiprocessingParallelSplitWriter:
    """Ray-free alternative to RayParallelSplitWriter: split the raw data
    into fixed-size chunks up front, then map a self-contained "write this
    chunk to its own shard" function over them with Pool.imap_unordered."""

    def __init__(self, num_workers: int = 4, chunk_size: int = 100) -> None:
        self.num_workers = num_workers
        self.chunk_size = chunk_size

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

        index_queue = mp.Queue()
        for i in range(self.num_workers):
            index_queue.put(i)

        total = len(dataset) if isinstance(dataset, Sequence) else None

        with mp.Pool(
            processes=self.num_workers,
            initializer=_init_mp_worker,
            initargs=(index_queue, transform, split_dir),
        ) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(_write_shard, enumerate(dataset)),
                total=total,
                desc=f"Writing split {split_name}",
            ):
                pass

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

        error_count = 0
        for idx, raw_item in tqdm.tqdm(
            enumerate(dataset), desc=f"Writing split {split_name}"
        ):
            try:
                writer.write(idx, raw_item)
            except DuplicateKeyError:
                logger.error(f"Duplicate key at sample index {idx}. Skipping.")
            except Exception:
                logger.exception(f"Error writing sample at index {idx}")
                error_count += 1

            if error_count >= 10:
                logger.error("Too many errors encountered. Stopping writer.")
                raise RuntimeError("Too many errors in SingleSplitWriter")

        write_info = writer.close()
        return self._finalize_shards(write_info)

    def _finalize_shards(
        self, write_info: list[DatasetShardInfo]
    ) -> list[DatasetShardInfo]:
        write_info = [x for x in write_info if x.nsamples > 0]
        return [replace(shard, shard=i + 1) for i, shard in enumerate(write_info)]


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
    ) -> None:
        self.max_shard_size = max_shard_size
        super().__init__(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
            use_ray=use_ray,
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

        dataset = split_iterator._dataset
        transform = split_iterator._transform

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
            writer = MultiprocessingParallelSplitWriter(
                num_workers=self.num_processes,
                chunk_size=min(self.max_shard_size, 1000),
            )
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
