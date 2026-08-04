from __future__ import annotations

import pickle

import pytest

from atria_core.datasets import Compose, IndexableSplitIterator, IterableSplitIterator


def test_indexable_split_iterator_getitem_and_len() -> None:
    iterator = IndexableSplitIterator(base_iterator=[0, 1, 2], transform=str)

    assert len(iterator) == 3
    assert iterator[0] == "0"
    assert iterator[2] == "2"


def test_indexable_split_iterator_getitems() -> None:
    iterator = IndexableSplitIterator(base_iterator=[0, 1, 2], transform=str)

    assert iterator.__getitems__([0, 2]) == ["0", "2"]


def test_indexable_split_iterator_iterates() -> None:
    iterator = IndexableSplitIterator(
        base_iterator=[0, 1, 2], transform=lambda x: x * 2
    )

    assert list(iterator) == [0, 2, 4]


def test_iterable_split_iterator_iterates() -> None:
    def _gen() -> object:
        yield from range(3)

    iterator = IterableSplitIterator(base_iterator=_gen(), transform=lambda x: x * 2)

    assert list(iterator) == [0, 2, 4]


def test_iterable_split_iterator_does_not_support_len_or_indexing() -> None:
    iterator = IterableSplitIterator(
        base_iterator=iter([1, 2, 3]), transform=lambda x: x
    )

    with pytest.raises(TypeError):
        len(iterator)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        iterator[0]  # type: ignore[index]


def test_with_transform_composes_without_mutating_original() -> None:
    base = IndexableSplitIterator(base_iterator=[1, 2, 3], transform=lambda x: x)
    doubled = base.with_transform(lambda x: x * 2)

    assert list(base) == [1, 2, 3]
    assert list(doubled) == [2, 4, 6]
    assert base is not doubled


def test_with_transform_chains_multiple_times() -> None:
    base = IndexableSplitIterator(base_iterator=[1, 2, 3], transform=lambda x: x)
    chained = base.with_transform(lambda x: x + 1).with_transform(lambda x: x * 10)

    assert list(chained) == [20, 30, 40]


def test_limit_returns_bounded_view_without_mutating_original() -> None:
    iterator = IndexableSplitIterator(
        base_iterator=[0, 1, 2, 3], transform=lambda value: value * 2
    )

    limited = iterator.limit(2)

    assert list(limited) == [0, 2]
    assert len(limited) == 2
    assert len(iterator) == 4


def test_compose_is_picklable() -> None:
    composed = Compose(str, len)
    restored = pickle.loads(pickle.dumps(composed))

    assert restored(123) == 3
