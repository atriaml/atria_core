from __future__ import annotations

from pathlib import Path

from atria_core.datasets import FileStorageType, IndexableSplitIterator
from atria_core.datasets._storage._storage_manager import StorageManager
from atria_core.types import DatasetSplitType
from atria_core.types._data_instance._base import BaseDataInstance


class _Record(BaseDataInstance):
    def to_dict(self) -> dict[str, object]:
        return {"sample_id": self.sample_id}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> _Record:
        return cls(sample_id=str(data["sample_id"]))


def _record(item: object) -> _Record:
    assert isinstance(item, _Record)
    return item


def test_msgpack_write_read_roundtrip(tmp_path: Path) -> None:
    records = [_Record(sample_id=str(i)) for i in range(5)]
    split_iterator = IndexableSplitIterator(
        base_iterator=records, transform=lambda x: x
    )

    storage_manager = StorageManager.create(
        FileStorageType.MSGPACK,
        data_dir=tmp_path,
        num_processes=1,
        storage_dir=tmp_path / "storage",
        config_name="test_config",
    )

    assert not storage_manager.split_exists(DatasetSplitType.train)
    storage_manager.write_split(DatasetSplitType.train, split_iterator)
    assert storage_manager.split_exists(DatasetSplitType.train)

    raw_read_back = storage_manager.read_split(DatasetSplitType.train)
    read_back = IndexableSplitIterator(
        base_iterator=raw_read_back, transform=_Record.from_dict
    )

    assert len(read_back) == 5
    assert {_record(sample).sample_id for sample in read_back} == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }


def test_purge_split_removes_written_data(tmp_path: Path) -> None:
    records = [_Record(sample_id="0")]
    split_iterator = IndexableSplitIterator(
        base_iterator=records, transform=lambda x: x
    )
    storage_manager = StorageManager.create(
        FileStorageType.MSGPACK,
        data_dir=tmp_path,
        num_processes=1,
        storage_dir=tmp_path / "storage",
        config_name="test_config",
    )

    storage_manager.write_split(DatasetSplitType.train, split_iterator)
    assert storage_manager.split_exists(DatasetSplitType.train)

    storage_manager.purge_split(DatasetSplitType.train)

    assert not storage_manager.split_exists(DatasetSplitType.train)
