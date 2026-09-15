from __future__ import annotations

import pickle

import pytest

from atria_core.datasets import (
    Compose,
    ConcatSplitIterator,
    IndexableSplitIterator,
    IterableSplitIterator,
)


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


def test_shuffle_returns_reordered_view_without_mutating_original() -> None:
    iterator = IndexableSplitIterator(
        base_iterator=list(range(10)), transform=lambda x: x
    )

    shuffled = iterator.shuffle(seed=1234)

    assert sorted(shuffled) == list(range(10))
    assert list(iterator) == list(range(10))
    assert list(shuffled) != list(range(10))


def test_shuffle_is_deterministic_for_a_given_seed() -> None:
    iterator = IndexableSplitIterator(
        base_iterator=list(range(20)), transform=lambda x: x
    )

    first = list(iterator.shuffle(seed=7))
    second = list(iterator.shuffle(seed=7))

    assert first == second


def test_shuffle_then_limit_samples_from_the_full_source() -> None:
    iterator = IndexableSplitIterator(
        base_iterator=list(range(100)), transform=lambda x: x
    )

    sampled = iterator.shuffle(seed=1).limit(5)

    assert len(sampled) == 5
    assert set(sampled) <= set(range(100))


def test_limit_then_shuffle_only_reorders_the_limited_view() -> None:
    iterator = IndexableSplitIterator(
        base_iterator=list(range(100)), transform=lambda x: x
    )

    reordered = iterator.limit(5).shuffle(seed=1)

    assert len(reordered) == 5
    assert set(reordered) <= set(range(5))


def test_concat_indexes_across_iterators_in_order() -> None:
    first = IndexableSplitIterator(base_iterator=[0, 1, 2], transform=lambda x: x)
    second = IndexableSplitIterator(base_iterator=[3, 4], transform=lambda x: x)

    concatenated = ConcatSplitIterator([first, second])

    assert len(concatenated) == 5
    assert list(concatenated) == [0, 1, 2, 3, 4]
    assert concatenated[0] == 0
    assert concatenated[4] == 4


def test_concat_getitem_out_of_range_raises() -> None:
    concatenated = ConcatSplitIterator([[0, 1], [2]])

    with pytest.raises(IndexError):
        concatenated[3]


def test_concat_composes_with_shuffled_and_limited_sources() -> None:
    first = (
        IndexableSplitIterator(base_iterator=list(range(10)), transform=lambda x: x)
        .shuffle(seed=1)
        .limit(3)
    )
    second = IndexableSplitIterator(
        base_iterator=list(range(100, 110)), transform=lambda x: x
    ).limit(2)

    concatenated = ConcatSplitIterator([first, second])

    assert len(concatenated) == 5
    assert set(concatenated[:3]) <= set(range(10))
    assert concatenated[3:] == [100, 101]


def test_compose_is_picklable() -> None:
    composed = Compose(str, len)
    restored = pickle.loads(pickle.dumps(composed))

    assert restored(123) == 3
