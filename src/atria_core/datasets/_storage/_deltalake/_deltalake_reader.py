from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from atria_core.datasets._common import DatasetLoadingMode
from atria_core.datasets._storage._deltalake._deltalake_row_writer import ParquetSchema


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
        """Return the reader implementing `mode`.

        Args:
            mode: How samples should be loaded.
            table_path: Delta table to read.
            storage_dir: Root the table lives under.
            config_name: Cache directory name.
            storage_options: Backend options passed to deltalake.

        Raises:
            NotImplementedError: If `mode` is online_streaming.
            ValueError: If `mode` is not a known loading mode.
        """
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
        """Return the whole split as a dataframe."""
        raise NotImplementedError


class InMemoryDeltalakeReader(DeltalakeReader):
    """Reads a Delta Lake split fully into memory."""

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
        """Return the whole split as a dataframe."""
        return self._df


class LocalDeltalakeReader(DeltalakeReader):
    """Reads a Delta Lake split lazily from local files."""

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
        """Return the whole split as a dataframe."""
        return self._dataset.to_table().to_pandas()
