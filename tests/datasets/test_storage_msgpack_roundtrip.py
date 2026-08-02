from __future__ import annotations

from pathlib import Path
from typing import Any

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


class _ListSplitIterator(IndexableSplitIterator[_Record]):
    def __init__(self, items: list[_Record], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._items = items

    def _raw_getitem(self, index: int) -> Any:
        return self._items[index]

    def _raw_len(self) -> int:
        return len(self._items)


def _record(item: object) -> _Record:
    assert isinstance(item, _Record)
    return item


def test_msgpack_write_read_roundtrip(tmp_path: Path) -> None:
    records = [_Record(sample_id=str(i)) for i in range(5)]
    split_iterator = _ListSplitIterator(
        items=records, split=DatasetSplitType.train, data_model=_Record
    )

    storage_manager = StorageManager.create(
        FileStorageType.MSGPACK,
        data_dir=tmp_path,
        num_processes=1,
        storage_dir=tmp_path / "storage",
        config_name="test_config",
    )

    assert not storage_manager.split_exists(DatasetSplitType.train)
    storage_manager.write_split(split_iterator=split_iterator)
    assert storage_manager.split_exists(DatasetSplitType.train)

    read_back = storage_manager.read_split(
        split=DatasetSplitType.train, data_model=_Record
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
    split_iterator = _ListSplitIterator(
        items=records, split=DatasetSplitType.train, data_model=_Record
    )
    storage_manager = StorageManager.create(
        FileStorageType.MSGPACK,
        data_dir=tmp_path,
        num_processes=1,
        storage_dir=tmp_path / "storage",
        config_name="test_config",
    )

    storage_manager.write_split(split_iterator=split_iterator)
    assert storage_manager.split_exists(DatasetSplitType.train)

    storage_manager.purge_split(DatasetSplitType.train)

    assert not storage_manager.split_exists(DatasetSplitType.train)
