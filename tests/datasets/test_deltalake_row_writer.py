from __future__ import annotations

from pathlib import Path

import deltalake

from atria_core.datasets._storage._deltalake._deltalake_row_writer import (
    _write_rows_to_deltalake,
)


def test_append_merges_columns_discovered_in_later_batch(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"

    _write_rows_to_deltalake([{"sample_id": "first"}], split_dir)
    _write_rows_to_deltalake(
        [{"sample_id": "second", "optional_text": "present"}],
        split_dir,
        mode="append",
    )

    rows = deltalake.DeltaTable(split_dir).to_pyarrow_table().to_pylist()
    assert sorted(rows, key=lambda row: row["sample_id"]) == [
        {"sample_id": "first", "optional_text": None},
        {"sample_id": "second", "optional_text": "present"},
    ]
