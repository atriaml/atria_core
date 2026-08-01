from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atria_core.serialization._artifact_store import ArtifactStore
from atria_core.serialization._row_codec import RowCodec
from atria_core.types import BaseDataInstance


class DatasetWriter:
    """Writes an iterable of same-typed BaseDataInstance objects to a
    directory: a single data.parquet (sample_id, type, data_json columns)
    plus an artifacts/ subdirectory holding any image/PDF bytes referenced
    by path. Single dataset-type per directory; no sharding, no compression
    tuning, no schema evolution -- see atria_datasets for that."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.directory)

    def write(self, instances: Iterable[BaseDataInstance]) -> None:
        rows = [RowCodec.to_row(instance, self.store) for instance in instances]
        table = pa.table(
            {
                "sample_id": [row["sample_id"] for row in rows],
                "type": [row["type"] for row in rows],
                "data_json": [row["data_json"] for row in rows],
            }
        )
        pq.write_table(table, self.directory / "data.parquet")
