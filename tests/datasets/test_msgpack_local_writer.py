from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from atria_core.datasets._storage._msgpack._msgpack_local_writer import (
    MultiprocessingParallelSplitWriter,
)
from atria_core.datasets._storage._msgpack._msgpack_shard_list_dataset import (
    MsgpackShardListDataset,
)


class _RecordDataclass:
    """Minimal stand-in for a DataInstance: `.to_dict()` and `.key` are all
    ShardWriterWorker.write() needs."""

    def __init__(self, index: int) -> None:
        self.index = index

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index}

    @property
    def key(self) -> str:
        return str(self.index)


class _VariableDelayTransform:
    """Makes each write take a random amount of time, so workers finish
    their share of the work at different moments -- the condition that lets
    a Pool's task scheduler skip a busy worker when finish-tasks are handed
    out (see MultiprocessingParallelSplitWriter's docstring)."""

    def __call__(self, index: int) -> _RecordDataclass:
        time.sleep(random.uniform(0, 0.003))
        return _RecordDataclass(index)


def test_every_worker_shard_is_fully_indexed_and_readable(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    num_workers = 8
    num_samples = 400

    writer = MultiprocessingParallelSplitWriter(num_workers=num_workers)
    shard_infos = writer.write_split(
        dataset=range(num_samples),
        transform=_VariableDelayTransform(),
        split_dir=split_dir,
    )

    assert len(shard_infos) == num_workers

    shard_files = sorted(split_dir.glob("*.msgpack"))
    assert len(shard_files) == num_workers

    for shard_file in shard_files:
        assert shard_file.with_suffix(".msgpack.offsets").exists(), (
            f"{shard_file.name} has no .offsets index -- its worker never closed it"
        )

    dataset = MsgpackShardListDataset(shard_files)
    assert len(dataset) == num_samples

    seen_indices = {record["index"] for record in dataset}
    assert seen_indices == set(range(num_samples))
