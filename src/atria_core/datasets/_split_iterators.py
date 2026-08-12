from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import islice
from typing import Any, Generic, TypeVar, overload

T_Output = TypeVar("T_Output")
T_NewOutput = TypeVar("T_NewOutput")


class Compose:
    """Plain class instead of a closure so composed transforms stay
    picklable -- stdlib pickle can't serialize nested functions, and
    multiprocessing writers need to send composed transforms to worker
    processes. Holds a flat list of transforms, flattening nested Compose
    instances on construction."""

    def __init__(self, *transforms: Callable[[Any], Any]) -> None:
        self.transforms: list[Callable[[Any], Any]] = []
        for t in transforms:
            self.transforms.extend(t.transforms if isinstance(t, Compose) else [t])

    def __call__(self, value: Any) -> Any:
        for transform in self.transforms:
            value = transform(value)
        return value


class IndexableSplitIterator(Sequence[T_Output], Generic[T_Output]):
    def __init__(
        self,
        base_iterator: Sequence[Any],
        transform: Callable[[Any], T_Output],
        max_samples: int | None = None,
    ) -> None:
        self._base_iterator = base_iterator
        self._transform = transform
        self._max_samples = max_samples

    def __len__(self) -> int:
        if self._max_samples is None:
            return len(self._base_iterator)
        return min(self._max_samples, len(self._base_iterator))

    @overload
    def __getitem__(self, index: int) -> T_Output: ...

    @overload
    def __getitem__(self, index: slice) -> list[T_Output]: ...

    def __getitem__(self, index: int | slice) -> T_Output | list[T_Output]:
        if isinstance(index, slice):
            return [self[item] for item in range(len(self))[index]]
        bounded_index = range(len(self))[index]
        return self._transform(self._base_iterator[bounded_index])

    def __getitems__(self, indices: list[int]) -> list[T_Output]:
        return [self[i] for i in indices]

    def with_transform(
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IndexableSplitIterator[T_NewOutput]:
        composed_transform = Compose(self._transform, transform)
        return IndexableSplitIterator(
            self._base_iterator,
            composed_transform,
            max_samples=self._max_samples,
        )

    def limit(self, max_samples: int) -> IndexableSplitIterator[T_Output]:
        if max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        return IndexableSplitIterator(
            self._base_iterator,
            self._transform,
            max_samples=min(max_samples, len(self)),
        )

    @property
    def base_iterator(self) -> Iterable[Any]:
        if self._max_samples is None:
            return self._base_iterator
        return islice(self._base_iterator, self._max_samples)

    @property
    def transform(self) -> Callable[[Any], T_Output]:
        return self._transform

    def __repr__(self) -> str:
        return f"IndexableSplitIterator(len={len(self)})"


class IterableSplitIterator(Iterable[T_Output], Generic[T_Output]):
    def __init__(
        self,
        base_iterator: Iterable[Any],
        transform: Callable[[Any], T_Output],
    ) -> None:
        self._base_iterator = base_iterator
        self._transform = transform

    def __iter__(self) -> Iterator[T_Output]:
        yield from map(self._transform, self._base_iterator)

    @property
    def base_iterator(self) -> Iterable[Any]:
        return self._base_iterator

    @property
    def transform(self) -> Callable[[Any], T_Output]:
        return self._transform

    def with_transform(
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IterableSplitIterator[T_NewOutput]:
        composed_transform = Compose(self._transform, transform)
        return IterableSplitIterator(self._base_iterator, composed_transform)

    def __repr__(self) -> str:
        return f"IterableSplitIterator({self._base_iterator})"
