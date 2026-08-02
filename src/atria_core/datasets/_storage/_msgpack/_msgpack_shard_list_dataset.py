from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from datadings.reader import MsgpackReader as MsgpackFileReader

from atria_core.logger import get_logger

logger = get_logger(__name__)


class MsgpackShardListDataset(Sequence[Any]):
    """Reads a dataset split back from a set of msgpack shard files, with
    efficient indexing across shards via cumulative sizes."""

    def __init__(self, shard_files: list[Path]) -> None:
        self._shard_files = shard_files
        self._shard_file_readers = [MsgpackFileReader(f) for f in shard_files]
        self._total_size: int = 0

        cumulative_sizes: list[int] = []
        for data in self._shard_file_readers:
            self._total_size += len(data)
            cumulative_sizes.append(self._total_size)
            data._close()
        self._cumulative_sizes = np.array(cumulative_sizes)

    @property
    def shard_files(self) -> list[Path]:
        return self._shard_files

    def fetch_sample_by_id(self, sample_id: str) -> tuple[int, dict[str, Any]]:
        for reader in self._shard_file_readers:
            try:
                sample_id = str(sample_id)
                index = reader.find_index(sample_id.replace(".", "_"))
                sample = reader[index]
                sample.pop("key", None)
                assert (
                    sample["sample_id"] == sample_id
                ), (  # this should never be triggered
                    f"Sample ID mismatch: expected {sample_id}, found {sample['sample_id']}"
                )
                return index, sample
            except KeyError:
                continue
        raise ValueError(f"Sample ID {sample_id} not found in any shard.")

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        shard_index = np.searchsorted(self._cumulative_sizes, index, side="right")
        if shard_index == 0:
            inner_index = index
        else:
            inner_index = index - self._cumulative_sizes[shard_index - 1]
        sample = self._shard_file_readers[shard_index][inner_index]
        sample.pop("key", None)
        return cast(dict[str, Any], sample)

    def __len__(self) -> int:
        return self._total_size

    def close(self) -> None:
        for reader in self._shard_file_readers:
            reader._close()
