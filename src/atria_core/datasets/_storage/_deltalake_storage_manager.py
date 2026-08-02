from __future__ import annotations

import itertools
import pickle
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

import deltalake
import pandas as pd
import pyarrow as pa
import ray
import tqdm

from atria_core.datasets._common import DatasetLoadingMode, T_BaseDataInstance
from atria_core.datasets._split_iterators import InstanceTransform, SplitIterator
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.logger import get_logger
from atria_core.serialization._artifact_store import ArtifactStore
from atria_core.serialization._row_codec import RowCodec
from atria_core.types import BaseDataInstance, DatasetSplitType

logger = get_logger(__name__)

_RAY_RUNTIME_ENV = {"env_vars": {"PYTHONPATH": ":".join(sys.path)}}


class DeltalakeWriterWorker:
    """Converts samples to parquet rows via `RowCodec`, materializing any
    image/PDF content into an `ArtifactStore` rooted under the split's
    write directory along the way."""

    def __init__(
        self,
        worker_id: int,
        write_dir: Path,
        data_dir: str | Path,
        split: str,
        data_model: type[BaseDataInstance],
        preprocess_transform: InstanceTransform[BaseDataInstance] | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._write_dir = Path(write_dir)
        self._data_dir = Path(data_dir)
        self._split = split
        self._data_model = data_model
        self._preprocess_transform = preprocess_transform
        self._store = ArtifactStore(self._write_dir / "data" / self._split)

    def load(self) -> Self:
        return self

    def write(self, index: int, sample: Any) -> list[dict[str, Any]]:
        if self._preprocess_transform is not None:
            result = self._preprocess_transform(index, sample)
            list_of_samples = result if isinstance(result, list) else [result]
        else:
            list_of_samples = [sample]
        return [RowCodec.to_row(s, self._store) for s in list_of_samples]

    def close(self) -> None:
        pass


@ray.remote
class DeltalakeShardWriterActor:
    def __init__(
        self,
        worker_id: int,
        write_dir: Path,
        data_dir: str | Path,
        split: str,
        data_model: type[BaseDataInstance],
        preprocess_transform: InstanceTransform[BaseDataInstance] | None = None,
    ) -> None:
        self.writer = DeltalakeWriterWorker(
            worker_id=worker_id,
            write_dir=write_dir,
            data_dir=data_dir,
            split=split,
            data_model=data_model,
            preprocess_transform=preprocess_transform,
        ).load()

    def write(self, sample_tuple: tuple[int, Any]) -> list[dict[str, Any]]:
        idx, sample = sample_tuple
        try:
            return self.writer.write(idx, sample)
        except Exception:
            logger.exception(f"Error writing sample at index {idx}")
            return []

    def close(self) -> None:
        self.writer.close()


def _write_rows_to_deltalake(
    rows: list[dict[str, Any]], split_dir: Path, mode: str = "overwrite"
) -> None:
    table = pa.table(
        {
            "sample_id": [row["sample_id"] for row in rows],
            "type": [row["type"] for row in rows],
            "data_json": [row["data_json"] for row in rows],
        }
    )
    deltalake.write_deltalake(str(split_dir), table, mode=mode)  # type: ignore[call-overload]


class RayParallelDeltalakeWriter:
    def __init__(
        self,
        write_dir: Path,
        data_dir: str | Path,
        num_workers: int = 4,
        max_concurrent_tasks_limit: int = 128,
        max_memory_per_actor: int = 500 * 1024 * 1024,
        max_memory: int = 1_000_000_000,
    ) -> None:
        self.write_dir = write_dir
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.max_concurrent_tasks_limit = max_concurrent_tasks_limit
        self.max_memory_per_actor = max_memory_per_actor
        self.max_memory = max_memory

    def write_split(self, split_iterator: SplitIterator[Any], split_dir: Path) -> None:
        split_name = split_iterator.split.value
        logger.info(f"Writing split {split_name} with {self.num_workers} Ray actors...")

        ray.init(
            num_cpus=self.num_workers,
            local_mode=self.num_workers == 1,
            runtime_env=_RAY_RUNTIME_ENV,
        )

        actors = [
            DeltalakeShardWriterActor.options(  # type: ignore[attr-defined]
                memory=self.max_memory_per_actor
            ).remote(
                worker_id=i,
                write_dir=self.write_dir,
                data_dir=self.data_dir,
                split=split_name,
                data_model=split_iterator.data_model,
                preprocess_transform=split_iterator._tf,
            )
            for i in range(self.num_workers)
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
            data_iterator = iter(split_iterator)
            pending_tasks = []
            actor_iterator = itertools.cycle(actors)

            for idx, sample in tqdm.tqdm(
                data_iterator, desc=f"Writing split {split_name}"
            ):
                actor = next(actor_iterator)
                pending_tasks.append(actor.write.remote((idx, sample)))

                if len(pending_tasks) >= self.max_concurrent_tasks_limit:
                    ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)
                    try:
                        for row_list in ray.get(ready_tasks):
                            coordinator_batch.extend(row_list)
                    except Exception as e:
                        logger.exception("Error in shard writer actor task")
                        raise e
                    _maybe_flush_batch()

            for row_list in ray.get(pending_tasks):
                coordinator_batch.extend(row_list)
            _maybe_flush_batch()

            ray.get([actor.close.remote() for actor in actors])

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


class SingleDeltalakeWriter:
    def __init__(
        self,
        data_dir: str | Path,
        write_dir: Path,
        max_memory: int = 1_000_000_000,
    ) -> None:
        self.data_dir = data_dir
        self.write_dir = write_dir
        self.max_memory = max_memory

    def write_split(self, split_iterator: SplitIterator[Any], split_dir: Path) -> None:
        split_name = split_iterator.split.value
        data_iterator = iter(split_iterator)

        worker = DeltalakeWriterWorker(
            worker_id=0,
            data_dir=self.data_dir,
            write_dir=self.write_dir,
            split=split_name,
            data_model=split_iterator.data_model,
            preprocess_transform=split_iterator._tf,
        ).load()

        coordinator_batch: list[dict[str, Any]] = []
        write_batch_size: int | None = None
        first_batch = True

        try:
            for idx, sample in tqdm.tqdm(
                data_iterator, desc=f"Writing split {split_name}"
            ):
                coordinator_batch.extend(worker.write(idx, sample))

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

        worker.close()

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
    ) -> None:
        self.max_memory = max_memory
        super().__init__(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
        )

    def split_exists(self, split: DatasetSplitType) -> bool:
        return (self.split_dir(split) / "_delta_log").exists()

    def _write_split_internal(self, split_iterator: SplitIterator[Any]) -> None:
        split_dir = self.split_dir(split_iterator.split)
        write_dir = self.storage_dir / self.config_name
        logger.info(
            f"Writing dataset split {split_iterator.split.value} to {split_dir} "
            f"({'parallel' if self.num_processes > 1 else 'single'} mode)"
        )

        writer: RayParallelDeltalakeWriter | SingleDeltalakeWriter
        if self.num_processes > 1:
            writer = RayParallelDeltalakeWriter(
                data_dir=self.data_dir,
                write_dir=write_dir,
                num_workers=self.num_processes,
                max_memory=self.max_memory,
            )
        else:
            writer = SingleDeltalakeWriter(
                data_dir=self.data_dir, write_dir=write_dir, max_memory=self.max_memory
            )

        split_iterator.disable_tf()
        try:
            writer.write_split(split_iterator, split_dir)
        finally:
            split_iterator.enable_tf()

    def read_split(
        self,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
        output_transform: Callable[
            [BaseDataInstance], BaseDataInstance | list[BaseDataInstance]
        ]
        | None = None,
        allowed_keys: set[str] | None = None,
        streaming_mode: bool = False,
    ) -> SplitIterator[Any]:
        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        if allowed_keys is not None:
            allowed_keys = allowed_keys.copy()
            allowed_keys.update({"sample_id", "index"})

        reader_cls = LocalDeltalakeReader if streaming_mode else InMemoryDeltalakeReader
        base_iterator: DeltalakeReader[BaseDataInstance] = reader_cls(
            table_path=str(self.split_dir(split=split)),
            data_model=data_model,
            allowed_keys=allowed_keys,
            storage_dir=str(self.storage_dir),
            config_name=self.config_name,
        )

        return SplitIterator(
            split=split,
            base_iterator=base_iterator,
            output_transform=output_transform,
            data_model=data_model,
        )

    def prepare_split_files(self, data_dir: str) -> set[tuple[str, str]]:
        delta_files = list((self.storage_dir / self.config_name).glob("**/*.*"))
        files_src_tgt = {
            (str(f), str(f.relative_to(self.storage_dir)))
            for f in delta_files
            if f.is_file()
        }

        def map_file_path(file_path: str) -> str | None:
            if pd.isna(file_path):
                return None
            parsed = urlparse(file_path)
            path = parsed.path

            if path.startswith("shards/"):
                tgt = str(Path(self.config_name) / path)
                files_src_tgt.add(
                    (str(Path(self.storage_dir) / self.config_name / path), tgt)
                )
            else:
                tgt = str(Path(self.config_name) / path)
                files_src_tgt.add((str(Path(data_dir) / path), tgt))
            return tgt

        for split in list(DatasetSplitType):
            if not self.split_exists(split):
                continue

            dt = deltalake.DeltaTable(self.split_dir(split=split))
            all_columns = [f.name for f in dt.schema().fields]

            file_path_cols = [col for col in all_columns if "file_path" in col.lower()]
            content_cols = [
                col.replace("file_path", "content") for col in file_path_cols
            ]
            dataframe = dt.to_pandas(columns=content_cols + file_path_cols)
            for file_path_col, content_col in zip(
                file_path_cols, content_cols, strict=True
            ):
                if not dataframe[content_col].dropna().empty:
                    continue
                file_path_col_data = dataframe[file_path_col].dropna()
                if file_path_col_data.empty:
                    continue
                file_path_col_data.apply(map_file_path)

        return files_src_tgt


class DeltalakeReader(Sequence[T_BaseDataInstance]):
    """Reads rows written by `RowCodec.to_row`/`ArtifactStore` back into
    data instances. Unlike the old per-field-flattened delta schema, rows
    here are always the fixed (sample_id, type, data_json) triple that
    `RowCodec` produces, and any binary content it materialized is already
    referenced by absolute path inside `data_json` -- so, unlike the old
    reader, there is no relative-path rewriting or column-level
    `allowed_keys` filtering to do; every row is decoded in full via
    `RowCodec.from_row`.
    """

    def __init__(
        self,
        table_path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = table_path
        self.data_model = data_model
        self.allowed_keys = allowed_keys
        self._length = 0

    @classmethod
    def from_mode(
        cls,
        mode: DatasetLoadingMode,
        table_path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
        storage_dir: str | None = None,
        config_path: str | None = None,
        storage_options: dict[str, Any] | None = None,
        presign_expiry: int = 3600,
    ) -> DeltalakeReader[T_BaseDataInstance]:
        kwargs = {
            "table_path": table_path,
            "data_model": data_model,
            "allowed_keys": allowed_keys,
            "storage_dir": storage_dir,
            "config_path": config_path,
            "storage_options": storage_options,
            "presign_expiry": presign_expiry,
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

    def __getitem__(self, index: int) -> T_BaseDataInstance:  # type: ignore[override]
        if isinstance(index, list):
            return self.__getitems__(index)
        return self._load_and_process_rows([index])[0]

    def __getitems__(self, indices: list[int]) -> list[T_BaseDataInstance]:
        return self._load_and_process_rows(indices)

    def _process_row(self, row: dict[str, Any]) -> T_BaseDataInstance:
        return RowCodec.from_row(row)  # type: ignore[return-value]

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        raise NotImplementedError

    def dataframe(self) -> pd.DataFrame:
        raise NotImplementedError


class InMemoryDeltalakeReader(DeltalakeReader[T_BaseDataInstance]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from deltalake import DeltaTable

        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        self.storage_options = kwargs.pop("storage_options", None)
        super().__init__(*args, **kwargs)
        self._df = DeltaTable(
            self.path, storage_options=self.storage_options
        ).to_pandas()
        self._length = len(self._df)

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        rows = self._df.iloc[indices]
        row_dicts = rows.to_dict(orient="records")
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._df


class LocalDeltalakeReader(DeltalakeReader[T_BaseDataInstance]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import pyarrow.dataset as ds
        from deltalake import DeltaTable
        from pyarrow.fs import S3FileSystem

        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        self.storage_options = kwargs.pop("storage_options", None)
        super().__init__(*args, **kwargs)
        delta = DeltaTable(self.path, storage_options=self.storage_options)
        file_uris = delta.file_uris()
        if file_uris[0].startswith("lakefs://"):
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

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        row_dicts = self._dataset.take(indices).to_pylist()
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._dataset.to_table().to_pandas()
