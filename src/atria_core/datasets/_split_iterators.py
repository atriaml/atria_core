from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Generic, TypeVar

T_Output = TypeVar("T_Output")
T_NewOutput = TypeVar("T_NewOutput")


def compose(
    first: Callable[[Any], Any],
    second: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    def composed(value: Any) -> Any:
        return second(first(value))

    return composed


class IndexableSplitIterator(Sequence[T_Output], Generic[T_Output]):
    def __init__(
        self,
        dataset: Sequence[Any],
        transform: Callable[[Any], T_Output],
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> T_Output:
        return self._transform(self._dataset[index])

    def __getitems__(self, indices: list[int]) -> list[T_Output]:
        return [self[i] for i in indices]

    def with_transform(
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IndexableSplitIterator[T_NewOutput]:
        composed_transform = compose(self._transform, transform)
        return IndexableSplitIterator(self._dataset, composed_transform)


class IterableSplitIterator(Iterable[T_Output], Generic[T_Output]):
    def __init__(
        self,
        dataset: Iterable[Any],
        transform: Callable[[Any], T_Output],
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __iter__(self) -> Iterator[T_Output]:
        yield from map(self._transform, self._dataset)

    def with_transform(
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IterableSplitIterator[T_NewOutput]:
        composed_transform = compose(self._transform, transform)
        return IterableSplitIterator(self._dataset, composed_transform)
