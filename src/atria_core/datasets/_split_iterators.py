from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Generic, TypeVar

T_Output = TypeVar("T_Output")
T_NewOutput = TypeVar("T_NewOutput")


class Compose:
    """Plain class instead of a closure so composed transforms stay
    picklable -- stdlib pickle can't serialize nested functions, and
    multiprocessing writers need to send composed transforms to worker
    processes."""

    def __init__(
        self,
        first: Callable[[Any], Any],
        second: Callable[[Any], Any],
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, value: Any) -> Any:
        return self.second(self.first(value))


def compose(
    first: Callable[[Any], Any],
    second: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    return Compose(first, second)


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

    def __getitem__(self, index: int) -> T_Output:  # type: ignore[override]
        return self._transform(self._dataset[index])

    def __getitems__(self, indices: list[int]) -> list[T_Output]:
        return [self[i] for i in indices]

    def with_transform(
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IndexableSplitIterator[T_NewOutput]:
        composed_transform = compose(self._transform, transform)
        return IndexableSplitIterator(self._dataset, composed_transform)

    def __repr__(self) -> str:
        return f"IndexableSplitIterator(len={len(self)})"


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

    def __repr__(self) -> str:
        return f"IterableSplitIterator({self._dataset})"
