from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import pickle
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import deltalake
import pandas as pd
import pyarrow as pa
import ray
import tqdm

from atria_core.datasets._common import DatasetLoadingMode
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType

logger = get_logger(__name__)

_RAY_RUNTIME_ENV = {"env_vars": {"PYTHONPATH": ":".join(sys.path)}}


class ParquetSchema:
    """Flattens a sample's to_dict() output into real parquet columns
    instead of one opaque JSON blob -- the whole point of Delta/parquet
    over msgpack is columnar structure. No per-instance-type dispatch:
    just walks whatever dict shape to_dict() produces."""

    @staticmethod
    def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(ParquetSchema.flatten(value, full_key))
            elif isinstance(value, list | tuple):
                # Variable-length per sample -- doesn't fit a fixed column,
                # so JSON-encode as a pragmatic fallback.
                flat[full_key] = json.dumps(value)
            else:
                flat[full_key] = value
        return flat

    @staticmethod
    def unflatten(flat: dict[str, Any]) -> dict[str, Any]:
        nested: dict[str, Any] = {}
        for key, value in flat.items():
            # pandas represents missing/null values as float NaN (not None)
            # even in otherwise-nullable string columns -- normalize back,
            # since NaN is not None and would slip past "is not None" checks.
            if isinstance(value, float) and math.isnan(value):
                value = None
            if isinstance(value, str) and value.startswith("["):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    parsed = value
                if isinstance(parsed, list):
                    value = parsed
            parts = key.split(".")
            target = nested
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        return nested

    @staticmethod
    def infer_pa_type(value: Any) -> pa.DataType:
        if isinstance(value, bool):
            return pa.bool_()
        if isinstance(value, int):
            return pa.int64()
        if isinstance(value, float):
            return pa.float64()
        if isinstance(value, bytes):
            return pa.binary()
        return pa.string()

    @staticmethod
    def merge(flat_rows: list[dict[str, Any]]) -> pa.Schema:
        """Different samples of the same data_model class generally
        flatten to the same key set (same dataclass shape) except for
        None-vs-populated optional fields -- union the keys seen, infer
        each column's type from the first non-None value seen for it."""
        columns: dict[str, pa.DataType] = {}
        for row in flat_rows:
            for key, value in row.items():
                if key in columns and columns[key] != pa.string():
                    continue
                if value is None:
                    columns.setdefault(key, pa.string())
                    continue
                columns[key] = ParquetSchema.infer_pa_type(value)
        return pa.schema([pa.field(k, v, nullable=True) for k, v in columns.items()])


def _hoist_bytes(data: dict[str, Any], artifacts_dir: Path, prefix: str) -> dict[str, Any]:
    """Writes any raw bytes value out to its own file under artifacts_dir
    and replaces it with a path reference, so parquet columns don't hold
    huge binary blobs -- type-agnostic, no ArtifactStore object involved.
    `content_bytes` is renamed to `file_path` to match how Image/PdfPage's
    own from_dict() expects a materialized (unloaded) instance to look."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, bytes):
            path = artifacts_dir / f"{prefix}-{key}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            new_key = "file_path" if key == "content_bytes" else key
            result[new_key] = str(path)
        elif isinstance(value, dict):
            result[key] = _hoist_bytes(value, artifacts_dir, f"{prefix}-{key}")
        else:
            result[key] = value
    return result


def _sample_to_row(sample: Any, artifacts_dir: Path) -> dict[str, Any]:
    hoisted = _hoist_bytes(sample.to_dict(), artifacts_dir, sample.key)
    return ParquetSchema.flatten(hoisted)


def _write_rows_to_deltalake(
    flat_rows: list[dict[str, Any]], split_dir: Path, mode: str = "overwrite"
) -> None:
    schema = ParquetSchema.merge(flat_rows)
    columns: dict[str, list[Any]] = {field.name: [] for field in schema}
    for row in flat_rows:
        for field in schema:
            columns[field.name].append(row.get(field.name))
    table = pa.table(columns, schema=schema)
    deltalake.write_deltalake(str(split_dir), table, mode=mode)  # type: ignore[call-overload]


def _write_items(
    raw_item: Any, transform: Callable[[Any], Any] | None, artifacts_dir: Path
) -> list[dict[str, Any]]:
    result = transform(raw_item) if transform is not None else raw_item
    items = result if isinstance(result, list) else [result]
    return [_sample_to_row(item, artifacts_dir) for item in items]


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
            runtime_env=_RAY_RUNTIME_ENV,
        )

        actors = [
            DeltalakeShardWriterActor.options(  # type: ignore[attr-defined]
                memory=self.max_memory_per_actor
            ).remote(artifacts_dir=artifacts_dir, transform=transform)
            for _ in range(self.num_workers)
        ]

        coordinator_batch: list[dict[str, Any]] = []
        write_batch_size: int | None = None
        first_batch = True

        def _maybe_flush_batch() -> None:
            nonlocal first_batch, write_batch_size
            if not coordinator_batch:
                return
            if write_batch_size is None:
                write_batch_size = max(
                    1, self.max_memory // len(pickle.dumps(coordinator_batch[0]))
                )
                logger.info(
                    f"Delta lake write batch size: {write_batch_size} rows "
                    f"(max_memory={self.max_memory // 1_000_000} MB)"
                )
            if len(coordinator_batch) >= write_batch_size:
                mode = "overwrite" if first_batch else "append"
                logger.info(
                    f"Writing batch of {len(coordinator_batch)} rows to delta lake at {split_dir}"
                )
                _write_rows_to_deltalake(coordinator_batch, split_dir, mode=mode)
                first_batch = False
                coordinator_batch.clear()

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
                        coordinator_batch.extend(row_list)
                    _maybe_flush_batch()

            for row_list in ray.get(pending_tasks):
                coordinator_batch.extend(row_list)
            _maybe_flush_batch()

            if coordinator_batch:
                mode = "overwrite" if first_batch else "append"
                logger.info(
                    f"Writing final batch of {len(coordinator_batch)} rows to delta lake at {split_dir}"
                )
                _write_rows_to_deltalake(coordinator_batch, split_dir, mode=mode)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, shutting down Ray actors...")
            raise
        except Exception as e:
            logger.exception("Error during Ray parallel split writing")
            raise e
        finally:
            if ray.is_initialized():
                ray.shutdown()


def _chunked(iterator: Iterator[Any], size: int) -> Iterator[list[Any]]:
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


_mp_delta_worker_state: tuple[Path, Callable[[Any], Any] | None] | None = None


def _init_mp_delta_worker(
    artifacts_dir: Path, transform: Callable[[Any], Any] | None
) -> None:
    global _mp_delta_worker_state
    _mp_delta_worker_state = (artifacts_dir, transform)


def _convert_chunk_to_rows(chunk: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    """Runs in a worker process, dispatched via Pool.imap: converts a chunk
    of raw items to rows (applying `transform`, materializing artifact
    content along the way). The actual Delta Lake writes still happen
    sequentially in the main process -- Delta appends must be ordered,
    unlike msgpack's independent per-chunk shard files."""
    assert _mp_delta_worker_state is not None, "Worker was not initialized"
    artifacts_dir, transform = _mp_delta_worker_state
    rows: list[dict[str, Any]] = []
    for idx, raw_item in chunk:
        try:
            rows.extend(_write_items(raw_item, transform, artifacts_dir))
        except Exception:
            logger.exception(f"Error writing sample at index {idx}")
    return rows


class MultiprocessingParallelDeltalakeWriter:
    """Ray-free alternative to RayParallelDeltalakeWriter: workers only
    convert raw-item chunks into rows via Pool.imap; this process still
    does the actual Delta Lake writes, sequentially, in batches."""

    def __init__(
        self,
        num_workers: int = 4,
        chunk_size: int = 1000,
        max_memory: int = 1_000_000_000,
    ) -> None:
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.max_memory = max_memory

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
        artifacts_dir: Path,
    ) -> None:
        split_name = split_dir.name
        logger.info(
            f"Writing split {split_name} with {self.num_workers} multiprocessing workers..."
        )

        coordinator_batch: list[dict[str, Any]] = []
        write_batch_size: int | None = None
        first_batch = True

        def _maybe_flush_batch() -> None:
            nonlocal first_batch, write_batch_size
            if not coordinator_batch:
                return
            if write_batch_size is None:
                write_batch_size = max(
                    1, self.max_memory // len(pickle.dumps(coordinator_batch[0]))
                )
                logger.info(
                    f"Delta lake write batch size: {write_batch_size} rows "
                    f"(max_memory={self.max_memory // 1_000_000} MB)"
                )
            if len(coordinator_batch) >= write_batch_size:
                mode = "overwrite" if first_batch else "append"
                logger.info(
                    f"Writing batch of {len(coordinator_batch)} rows to delta lake at {split_dir}"
                )
                _write_rows_to_deltalake(coordinator_batch, split_dir, mode=mode)
                first_batch = False
                coordinator_batch.clear()

        chunks = _chunked(enumerate(dataset), self.chunk_size)
        with mp.Pool(
            processes=self.num_workers,
            initializer=_init_mp_delta_worker,
            initargs=(artifacts_dir, transform),
        ) as pool:
            for rows in tqdm.tqdm(
                pool.imap(_convert_chunk_to_rows, chunks),
                desc=f"Writing split {split_name}",
            ):
                coordinator_batch.extend(rows)
                _maybe_flush_batch()

        if coordinator_batch:
            mode = "overwrite" if first_batch else "append"
            logger.info(
                f"Writing final batch of {len(coordinator_batch)} rows to delta lake at {split_dir}"
            )
            _write_rows_to_deltalake(coordinator_batch, split_dir, mode=mode)


class SingleDeltalakeWriter:
    def __init__(self, max_memory: int = 1_000_000_000) -> None:
        self.max_memory = max_memory

    def write_split(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        transform: Callable[[Any], Any] | None,
        split_dir: Path,
        artifacts_dir: Path,
    ) -> None:
        split_name = split_dir.name

        coordinator_batch: list[dict[str, Any]] = []
        write_batch_size: int | None = None
        first_batch = True

        try:
            for idx, raw_item in tqdm.tqdm(
                enumerate(dataset), desc=f"Writing split {split_name}"
            ):
                try:
                    coordinator_batch.extend(
                        _write_items(raw_item, transform, artifacts_dir)
                    )
                except Exception:
                    logger.exception(f"Error writing sample at index {idx}")
                    continue

                if write_batch_size is None and coordinator_batch:
                    write_batch_size = max(
                        1, self.max_memory // len(pickle.dumps(coordinator_batch[0]))
                    )
                    logger.info(
                        f"Delta lake write batch size: {write_batch_size} rows "
                        f"(max_memory={self.max_memory // 1_000_000} MB)"
                    )

                if write_batch_size and len(coordinator_batch) >= write_batch_size:
                    mode = "overwrite" if first_batch else "append"
                    logger.info(
                        f"Writing batch of {len(coordinator_batch)} rows to delta lake at {split_dir}"
                    )
                    _write_rows_to_deltalake(coordinator_batch, split_dir, mode=mode)
                    first_batch = False
                    coordinator_batch.clear()

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, stopping split writing...")
            raise

        if coordinator_batch:
            mode = "overwrite" if first_batch else "append"
            logger.info(
                f"Writing final batch of {len(coordinator_batch)} rows to delta lake at {split_dir}"
            )
            _write_rows_to_deltalake(coordinator_batch, split_dir, mode=mode)


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
    ) -> None:
        self.max_memory = max_memory
        super().__init__(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
            use_ray=use_ray,
        )

    def split_exists(self, split: DatasetSplitType) -> bool:
        return (self.split_dir(split) / "_delta_log").exists()

    def _write_split_internal(self, split: DatasetSplitType, split_iterator: Any) -> None:
        split_dir = self.split_dir(split)
        write_dir = self.storage_dir / self.config_name
        artifacts_dir = write_dir / "data" / split.value / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Writing dataset split {split.value} to {split_dir} "
            f"({'parallel' if self.num_processes > 1 else 'single'} mode)"
        )

        dataset = split_iterator._dataset
        transform = split_iterator._transform

        writer: (
            RayParallelDeltalakeWriter
            | MultiprocessingParallelDeltalakeWriter
            | SingleDeltalakeWriter
        )
        if self.num_processes > 1 and self.use_ray:
            writer = RayParallelDeltalakeWriter(
                num_workers=self.num_processes, max_memory=self.max_memory
            )
            writer.write_split(dataset, transform, split_dir, artifacts_dir)
        elif self.num_processes > 1:
            writer = MultiprocessingParallelDeltalakeWriter(
                num_workers=self.num_processes, max_memory=self.max_memory
            )
            writer.write_split(dataset, transform, split_dir, artifacts_dir)
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


class DeltalakeReader(Sequence[Any]):
    """Reads rows written by `_sample_to_row` back into raw dicts (via
    ParquetSchema.unflatten) -- decoding into a data_model instance happens
    through Dataset's own generic input-transform wrapping, not here."""

    def __init__(
        self,
        table_path: str,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = table_path
        self.storage_options = storage_options
        self._length = 0

    @classmethod
    def from_mode(
        cls,
        mode: DatasetLoadingMode,
        table_path: str,
        storage_dir: str | None = None,
        config_name: str | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> DeltalakeReader:
        kwargs = {
            "table_path": table_path,
            "storage_dir": storage_dir,
            "config_name": config_name,
            "storage_options": storage_options,
        }
        if mode == DatasetLoadingMode.in_memory:
            return InMemoryDeltalakeReader(**kwargs)
        elif mode == DatasetLoadingMode.local_streaming:
            return LocalDeltalakeReader(**kwargs)
        elif mode == DatasetLoadingMode.online_streaming:
            # OnlineDeltalakeReader (presigned-S3/lakeFS streaming) is out of
            # scope for this port -- it depends on boto3/lakefs.
            raise NotImplementedError(
                "online_streaming loading mode is not supported in this build."
            )
        else:
            raise ValueError(f"Unsupported loading mode: {mode}")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Any:  # type: ignore[override]
        if isinstance(index, list):
            return self.__getitems__(index)
        return self._load_and_process_rows([index])[0]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        return self._load_and_process_rows(indices)

    def _process_row(self, row: dict[str, Any]) -> Any:
        return ParquetSchema.unflatten(row)

    def _load_and_process_rows(self, indices: list[int]) -> list[Any]:
        raise NotImplementedError

    def dataframe(self) -> pd.DataFrame:
        raise NotImplementedError


class InMemoryDeltalakeReader(DeltalakeReader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from deltalake import DeltaTable

        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        super().__init__(*args, **kwargs)
        self._df = DeltaTable(
            self.path, storage_options=self.storage_options
        ).to_pandas()
        self._length = len(self._df)

    def _load_and_process_rows(self, indices: list[int]) -> list[Any]:
        rows = self._df.iloc[indices]
        row_dicts = rows.to_dict(orient="records")
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._df


class LocalDeltalakeReader(DeltalakeReader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import pyarrow.dataset as ds
        from deltalake import DeltaTable
        from pyarrow.fs import S3FileSystem

        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        super().__init__(*args, **kwargs)
        delta = DeltaTable(self.path, storage_options=self.storage_options)
        file_uris = delta.file_uris()
        if file_uris[0].startswith("lakefs://"):
            assert self.storage_options is not None, (
                "storage_options is required for lakefs:// URIs."
            )
            # set AWS_EC2_METADATA_DISABLED to improve load time of s3fs see https://github.com/apache/arrow/issues/37136
            filesystem = S3FileSystem(
                endpoint_override=self.storage_options["AWS_ENDPOINT"],
                access_key=self.storage_options["AWS_ACCESS_KEY_ID"],
                secret_key=self.storage_options["AWS_SECRET_ACCESS_KEY"],
            )
            file_uris = [file_uri.replace("lakefs://", "") for file_uri in file_uris]
            self._dataset = ds.dataset(file_uris, filesystem=filesystem)
        else:
            self._dataset = ds.dataset(self.path)
        self._length = self._dataset.count_rows()

    def _load_and_process_rows(self, indices: list[int]) -> list[Any]:
        row_dicts = self._dataset.take(indices).to_pylist()
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._dataset.to_table().to_pandas()
