from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import islice
from typing import Any, overload

from atria_core.types._utilities._repr import RepresentationMixin


class Compose(RepresentationMixin):
    """Plain class instead of a closure so composed transforms stay
    picklable -- stdlib pickle can't serialize nested functions, and
    multiprocessing writers need to send composed transforms to worker
    processes. Holds a flat list of transforms, flattening nested Compose
    instances on construction."""

    __repr_fields__ = ("transforms",)

    def __init__(self, *transforms: Callable[[Any], Any]) -> None:
        self.transforms: list[Callable[[Any], Any]] = []
        for t in transforms:
            self.transforms.extend(t.transforms if isinstance(t, Compose) else [t])

    def __call__(self, value: Any) -> Any:
        for transform in self.transforms:
            value = transform(value)
        return value


class IndexableSplitIterator[T_Output](Sequence[T_Output], RepresentationMixin):
    """A random-access split: wraps an indexable raw source and applies a
    transform to each sample on access, optionally restricted to
    `mapped_indices` -- indices into the raw source, in the order these
    samples should be exposed in. A cap (`limit()`) and a reorder
    (`shuffle()`) are both just different ways of picking `mapped_indices`,
    so one field covers both."""

    __repr_fields__ = ("length", "base_iterator", "transform")

    def __init__(
        self,
        base_iterator: Sequence[Any],
        transform: Callable[[Any], T_Output],
        mapped_indices: Sequence[int] | None = None,
    ) -> None:
        self._base_iterator = base_iterator
        self._transform = transform
        self._mapped_indices = mapped_indices

    def __len__(self) -> int:
        if self._mapped_indices is not None:
            return len(self._mapped_indices)
        return len(self._base_iterator)

    def _resolve(self, index: int) -> int:
        return self._mapped_indices[index] if self._mapped_indices is not None else index

    @overload
    def __getitem__(self, index: int) -> T_Output: ...

    @overload
    def __getitem__(self, index: slice) -> list[T_Output]: ...

    def __getitem__(self, index: int | slice) -> T_Output | list[T_Output]:
        if isinstance(index, slice):
            return [self[item] for item in range(len(self))[index]]
        bounded_index = range(len(self))[index]
        return self._transform(self._base_iterator[self._resolve(bounded_index)])

    def __getitems__(self, indices: list[int]) -> list[T_Output]:
        return [self[i] for i in indices]

    def with_transform[T_NewOutput](
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IndexableSplitIterator[T_NewOutput]:
        """Return a new iterator applying `transform` after the current one."""
        composed_transform = Compose(self._transform, transform)
        return IndexableSplitIterator(
            base_iterator=self._base_iterator,
            transform=composed_transform,
            mapped_indices=self._mapped_indices,
        )

    def limit(self, max_samples: int) -> IndexableSplitIterator[T_Output]:
        """Return an iterator exposing at most `max_samples` of these samples."""
        if max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        indices = [self._resolve(i) for i in range(min(max_samples, len(self)))]
        return IndexableSplitIterator(
            base_iterator=self._base_iterator,
            transform=self._transform,
            mapped_indices=indices,
        )

    def shuffle(self, seed: int) -> IndexableSplitIterator[T_Output]:
        """Return an iterator exposing these samples in a shuffled order.

        Composes with `limit()` in either order: shuffle-then-limit takes a
        random sample of the source (the paper's `.shuffle(seed).take(n)`
        pattern); limit-then-shuffle instead reorders whatever was already
        selected.
        """
        indices = [self._resolve(i) for i in range(len(self))]
        random.Random(seed).shuffle(indices)
        return IndexableSplitIterator(
            base_iterator=self._base_iterator,
            transform=self._transform,
            mapped_indices=indices,
        )

    @property
    def base_iterator(self) -> Iterable[Any]:
        """The untransformed source, in the order these samples are exposed."""
        if self._mapped_indices is None:
            return self._base_iterator
        return (self._base_iterator[i] for i in self._mapped_indices)

    @property
    def transform(self) -> Callable[[Any], T_Output]:
        """The transform applied to each raw sample."""
        return self._transform

    @property
    def length(self) -> int:
        return len(self)

    @property
    def mapped_indices(self) -> Sequence[int] | None:
        """Indices into the raw source, in exposure order -- None means the
        source's own order and length, unrestricted."""
        return self._mapped_indices


class ConcatSplitIterator[T_Output](Sequence[T_Output], RepresentationMixin):
    """A random-access view over several sequences end to end -- e.g. several
    datasets' split iterators, each already shuffled/limited/transformed to
    a common output type. Indexing walks across them in order; `len()` is
    their combined length."""

    __repr_fields__ = ("length", "iterators")

    def __init__(self, iterators: Sequence[Sequence[T_Output]]) -> None:
        self._iterators = iterators

    def __len__(self) -> int:
        return sum(len(iterator) for iterator in self._iterators)

    @overload
    def __getitem__(self, index: int) -> T_Output: ...

    @overload
    def __getitem__(self, index: slice) -> list[T_Output]: ...

    def __getitem__(self, index: int | slice) -> T_Output | list[T_Output]:
        if isinstance(index, slice):
            return [self[item] for item in range(len(self))[index]]
        bounded_index = range(len(self))[index]
        for iterator in self._iterators:
            if bounded_index < len(iterator):
                return iterator[bounded_index]
            bounded_index -= len(iterator)
        raise IndexError(index)  # pragma: no cover -- unreachable, bounded_index is in range(len(self))

    def __getitems__(self, indices: list[int]) -> list[T_Output]:
        return [self[i] for i in indices]

    @property
    def iterators(self) -> Sequence[Sequence[T_Output]]:
        return self._iterators

    @property
    def length(self) -> int:
        return len(self)


class IterableSplitIterator[T_Output](Iterable[T_Output], RepresentationMixin):
    """A streaming split: wraps a one-pass raw source and applies a transform
    to each sample as it is yielded, optionally capped at `max_samples`."""

    __repr_fields__ = ("base_iterator", "transform", "max_samples")

    def __init__(
        self,
        base_iterator: Iterable[Any],
        transform: Callable[[Any], T_Output],
        max_samples: int | None = None,
    ) -> None:
        self._base_iterator = base_iterator
        self._transform = transform
        self._max_samples = max_samples

    def __iter__(self) -> Iterator[T_Output]:
        yield from map(self._transform, self.base_iterator)

    @property
    def base_iterator(self) -> Iterable[Any]:
        """The untransformed source, capped at `max_samples` if one is set."""
        if self._max_samples is None:
            return self._base_iterator
        return islice(self._base_iterator, self._max_samples)

    @property
    def transform(self) -> Callable[[Any], T_Output]:
        """The transform applied to each raw sample."""
        return self._transform

    def with_transform[T_NewOutput](
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IterableSplitIterator[T_NewOutput]:
        """Return a new iterator applying `transform` after the current one."""
        composed_transform = Compose(self._transform, transform)
        return IterableSplitIterator(
            base_iterator=self._base_iterator,
            transform=composed_transform,
            max_samples=self._max_samples,
        )

    def limit(self, max_samples: int) -> IterableSplitIterator[T_Output]:
        """Return an iterator yielding at most `max_samples` of these samples."""
        if max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        if self._max_samples is not None:
            max_samples = min(max_samples, self._max_samples)
        return IterableSplitIterator(
            base_iterator=self._base_iterator,
            transform=self._transform,
            max_samples=max_samples,
        )

    @property
    def max_samples(self) -> int | None:
        return self._max_samples
