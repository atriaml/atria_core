from __future__ import annotations

import json
import math
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import deltalake
import pyarrow as pa

from atria_core.logger import get_logger

logger = get_logger(__name__)

_INLINE_BYTES_LIMIT = 1_048_576  # 1 MiB (1024 * 1024)


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


def _hoist_bytes(
    data: dict[str, Any],
    artifacts_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    """Writes large `bytes` values out to artifact files.

    Small byte strings are kept inline. `content_bytes` is only materialized
    when it exceeds the inline limit and there is no existing `file_path`.
    If `file_path` already exists, it is reused instead of duplicating the
    content.
    """
    has_file_path = bool(data.get("file_path"))
    result: dict[str, Any] = {}

    for key, value in data.items():
        if key == "file_path" and value is None:
            # Superseded by the hoisted content_bytes path below, if any.
            continue

        if isinstance(value, bytes):
            if key == "content_bytes" and has_file_path:
                continue

            if len(value) <= _INLINE_BYTES_LIMIT:
                result[key] = value
                continue

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


class _DeltaBatchWriter:
    """Shared row-accumulation + periodic flush logic used by every Delta
    parallelism mode (Ray/multiprocessing/single): buffers converted rows
    and writes them out via `_write_rows_to_deltalake` once the buffer
    exceeds `max_memory` (estimated from the pickled size of the first
    row seen), alternating `overwrite` (first batch) then `append`."""

    def __init__(self, split_dir: Path, max_memory: int) -> None:
        self._split_dir = split_dir
        self._max_memory = max_memory
        self._batch: list[dict[str, Any]] = []
        self._write_batch_size: int | None = None
        self._first_batch = True

    def add(self, rows: list[dict[str, Any]]) -> None:
        self._batch.extend(rows)
        if self._write_batch_size is None and self._batch:
            self._write_batch_size = max(
                1, self._max_memory // len(pickle.dumps(self._batch[0]))
            )
            logger.info(
                f"Delta lake write batch size: {self._write_batch_size} rows "
                f"(max_memory={self._max_memory // 1_000_000} MB)"
            )
        if self._write_batch_size and len(self._batch) >= self._write_batch_size:
            self._flush()

    def finish(self) -> None:
        if self._batch:
            self._flush()

    def _flush(self) -> None:
        mode = "overwrite" if self._first_batch else "append"
        logger.info(
            f"Writing batch of {len(self._batch)} rows to delta lake at {self._split_dir}"
        )
        _write_rows_to_deltalake(self._batch, self._split_dir, mode=mode)
        self._first_batch = False
        self._batch.clear()
