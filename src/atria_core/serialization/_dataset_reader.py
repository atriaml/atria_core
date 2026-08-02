from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from atria_core.serialization._row_codec import RowCodec
from atria_core.types import BaseDataInstance


class DatasetReader:
    """Reads a directory written by DatasetWriter back into BaseDataInstance
    objects. Both ImageInstance's Image and SinglePageDocumentInstance's
    visual (Image or PdfPage) stay lazy -- content loads on .load()."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def __iter__(self) -> Iterator[BaseDataInstance]:
        table = pq.read_table(self.directory / "data.parquet")
        for row in table.to_pylist():
            yield RowCodec.from_row(row)
