from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Self

from datadings.writer import Writer

from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo

logger = get_logger(__name__)


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


def safe_write(writer: ShardWriterWorker, idx: int, raw_item: Any) -> None:
    try:
        writer.write(idx, raw_item)
    except DuplicateKeyError:
        logger.error(f"Duplicate key at index {idx}, skipping")
    except Exception:
        logger.exception(f"Error writing sample at index {idx}")
