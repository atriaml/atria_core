from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._serialization._row_codec import RowCodec


class DatasetReader:
    """Reads a directory written by DatasetWriter back into BaseDataInstance
    objects. ImageInstance's Image stays lazy (content loads on Image.load());
    DocumentInstance's SinglePageDocument loads its page image eagerly, same
    as SinglePageDocument.from_image does everywhere else in this library."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def __iter__(self) -> Iterator[BaseDataInstance]:
        table = pq.read_table(self.directory / "data.parquet")
        for row in table.to_pylist():
            yield RowCodec.from_row(row)
