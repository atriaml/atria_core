from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any

import pytest

from atria_core.datasets import (
    HFSplitIterator,
    IndexableSplitIterator,
    IterableSplitIterator,
)
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


class _GeneratorSplitIterator(IterableSplitIterator[_Record]):
    def __init__(self, items: Iterator[_Record], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._items = items

    def _raw_iter(self) -> Iterator[_Record]:
        return self._items


def _records(n: int) -> list[_Record]:
    return [_Record(sample_id=str(i)) for i in range(n)]


def _record(item: _Record | list[_Record] | tuple[int, _Record]) -> _Record:
    assert isinstance(item, _Record)
    return item


def test_iterates_over_indexable_split_iterator() -> None:
    iterator = _ListSplitIterator(
        items=_records(3), split=DatasetSplitType.train, data_model=_Record
    )

    result = [_record(item) for item in iterator]

    assert [r.sample_id for r in result] == ["0", "1", "2"]


def test_getitem_and_len() -> None:
    iterator = _ListSplitIterator(
        items=_records(3), split=DatasetSplitType.train, data_model=_Record
    )

    assert len(iterator) == 3
    assert _record(iterator[1]).sample_id == "1"


def test_max_len_truncates() -> None:
    iterator = _ListSplitIterator(
        items=_records(5),
        split=DatasetSplitType.train,
        data_model=_Record,
        max_len=2,
    )

    assert len(iterator) == 2
    assert len(list(iterator)) == 2


def test_subset_indices_restricts_access() -> None:
    iterator = _ListSplitIterator(
        items=_records(5), split=DatasetSplitType.train, data_model=_Record
    )
    iterator.subset_indices = [4, 2, 0]

    assert len(iterator) == 3
    assert [_record(iterator[i]).sample_id for i in range(3)] == ["4", "2", "0"]


def test_output_transform_applied() -> None:
    iterator = _ListSplitIterator(
        items=_records(2),
        split=DatasetSplitType.train,
        data_model=_Record,
        output_transform=lambda sample: _Record(
            sample_id=f"transformed-{sample.sample_id}"
        ),
    )

    assert _record(iterator[0]).sample_id == "transformed-0"


def test_disable_tf_yields_raw_index_sample_pairs() -> None:
    iterator = _ListSplitIterator(
        items=_records(2), split=DatasetSplitType.train, data_model=_Record
    )
    iterator.disable_tf()

    item = iterator[0]

    assert isinstance(item, tuple)
    index, sample = item
    assert index == 0
    assert sample.sample_id == "0"


def test_get_random_subset() -> None:
    iterator = _ListSplitIterator(
        items=_records(10), split=DatasetSplitType.train, data_model=_Record
    )

    subset = iterator.get_random_subset(subset_size=3, seed=1)

    assert len(subset) == 3
    assert subset is not iterator


def _record_generator() -> Generator[_Record, None, None]:
    yield from _records(2)


def test_iterable_split_iterator_does_not_support_indexing() -> None:
    iterator = _GeneratorSplitIterator(
        items=_record_generator(), split=DatasetSplitType.train, data_model=_Record
    )

    with pytest.raises(RuntimeError):
        _ = iterator[0]


def test_hf_split_iterator_iterates() -> None:
    iterator = HFSplitIterator(
        hf_dataset=_records(2), split=DatasetSplitType.train, data_model=_Record
    )

    assert [_record(item).sample_id for item in iterator] == ["0", "1"]


def test_hf_split_iterator_forces_iterable_only_mode() -> None:
    # hf_dataset here is a plain list (indexable), but HFSplitIterator
    # always forces iterable-only access regardless of what it wraps.
    iterator = HFSplitIterator(
        hf_dataset=_records(2), split=DatasetSplitType.train, data_model=_Record
    )

    with pytest.raises(RuntimeError):
        _ = iterator[0]
