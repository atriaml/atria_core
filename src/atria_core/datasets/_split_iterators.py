from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast, overload

import numpy as np

from atria_core.datasets._pandas import samples_to_pandas
from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._utilities._repr import RepresentationMixin

if TYPE_CHECKING:
    import pandas as pd


class Compose(RepresentationMixin):
    __repr_fields__: ClassVar[Collection[str]] = ("transforms",)

    def __init__(self, *transforms: Callable[[Any], Any]) -> None:
        self.transforms: list[Callable[[Any], Any]] = []
        for t in transforms:
            self.transforms.extend(t.transforms if isinstance(t, Compose) else [t])

    def __call__(self, value: Any) -> Any:
        for transform in self.transforms:
            value = transform(value)
        return value


class IndexableSplitIterator[T_Output](Sequence[T_Output], RepresentationMixin):
    __repr_fields__: ClassVar[Collection[str]] = (
        "length",
        "base_iterator",
        "transform",
    )

    def __init__(
        self,
        base_iterator: Sequence[Any],
        transform: Callable[[Any], T_Output] = lambda x: x,
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
        return (
            self._mapped_indices[index] if self._mapped_indices is not None else index
        )

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
        composed_transform = Compose(self._transform, transform)
        return IndexableSplitIterator(
            base_iterator=self._base_iterator,
            transform=composed_transform,
            mapped_indices=self._mapped_indices,
        )

    def limit(self, max_samples: int) -> IndexableSplitIterator[T_Output]:
        if max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        indices = [self._resolve(i) for i in range(min(max_samples, len(self)))]
        return IndexableSplitIterator(
            base_iterator=self._base_iterator,
            transform=self._transform,
            mapped_indices=indices,
        )

    def shuffle(self, seed: int) -> IndexableSplitIterator[T_Output]:
        base_indices = [self._resolve(i) for i in range(len(self))]
        permutation = np.random.default_rng(seed).permutation(len(base_indices))
        indices = [base_indices[int(i)] for i in permutation]
        return IndexableSplitIterator(
            base_iterator=self._base_iterator,
            transform=self._transform,
            mapped_indices=indices,
        )

    def filter(
        self, predicate: Callable[[T_Output], bool]
    ) -> IndexableSplitIterator[T_Output]:
        indices = [self._resolve(i) for i in range(len(self)) if predicate(self[i])]
        return IndexableSplitIterator(
            base_iterator=self._base_iterator,
            transform=self._transform,
            mapped_indices=indices,
        )

    @property
    def base_iterator(self) -> Iterable[Any]:
        if self._mapped_indices is None:
            return self._base_iterator
        return (self._base_iterator[i] for i in self._mapped_indices)

    @property
    def transform(self) -> Callable[[Any], T_Output]:
        return self._transform

    @property
    def length(self) -> int:
        return len(self)

    @property
    def mapped_indices(self) -> Sequence[int] | None:
        return self._mapped_indices

    def to_pandas(self) -> pd.DataFrame:
        return samples_to_pandas(self)

    def to_jsonl(self, path: Path) -> int:
        """Materialize the iterator outputs to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with path.open("w", encoding="utf-8") as output_file:
            for sample in self:
                if not isinstance(sample, BaseDataModel):
                    raise TypeError(
                        f"{type(sample).__name__} does not implement to_dict()"
                    )

                output_file.write(
                    json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
                )
                written += 1

        return written

    @classmethod
    def from_jsonl[T_DataModel: BaseDataModel](
        cls,
        path: Path,
        output_type: type[T_DataModel],
    ) -> IndexableSplitIterator[T_DataModel]:
        """Load materialized iterator outputs from a JSONL file."""
        samples: list[T_DataModel] = []

        with path.open(encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                if not line.strip():
                    continue

                try:
                    sample = cast(
                        "T_DataModel", output_type.from_dict(json.loads(line))
                    )
                    samples.append(sample)
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    raise ValueError(
                        f"Invalid {output_type.__name__} at {path}:{line_number}"
                    ) from error

        return cast(
            "IndexableSplitIterator[T_DataModel]",
            cls(base_iterator=samples, transform=lambda sample: sample),
        )


class _ConcatenatedSequence[T_Output](Sequence[T_Output]):
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
        raise IndexError(index)  # pragma: no cover -- unreachable


class ConcatSplitIterator[T_Output](IndexableSplitIterator[T_Output]):
    __repr_fields__: ClassVar[Collection[str]] = ("length", "iterators")

    def __init__(self, iterators: Sequence[Sequence[T_Output]]) -> None:
        super().__init__(
            base_iterator=_ConcatenatedSequence(iterators), transform=lambda x: x
        )
        self._iterators = iterators

    @property
    def iterators(self) -> Sequence[Sequence[T_Output]]:
        return self._iterators


class IterableSplitIterator[T_Output](Iterable[T_Output], RepresentationMixin):
    __repr_fields__: ClassVar[Collection[str]] = (
        "base_iterator",
        "transform",
        "max_samples",
    )

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
        if self._max_samples is None:
            return self._base_iterator
        return islice(self._base_iterator, self._max_samples)

    @property
    def transform(self) -> Callable[[Any], T_Output]:
        return self._transform

    def with_transform[T_NewOutput](
        self, transform: Callable[[T_Output], T_NewOutput]
    ) -> IterableSplitIterator[T_NewOutput]:
        composed_transform = Compose(self._transform, transform)
        return IterableSplitIterator(
            base_iterator=self._base_iterator,
            transform=composed_transform,
            max_samples=self._max_samples,
        )

    def limit(self, max_samples: int) -> IterableSplitIterator[T_Output]:
        if max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        if self._max_samples is not None:
            max_samples = min(max_samples, self._max_samples)
        return IterableSplitIterator(
            base_iterator=self._base_iterator,
            transform=self._transform,
            max_samples=max_samples,
        )

    def filter(
        self, predicate: Callable[[T_Output], bool]
    ) -> IterableSplitIterator[T_Output]:
        return IterableSplitIterator(
            base_iterator=filter(
                lambda raw: predicate(self._transform(raw)), self.base_iterator
            ),
            transform=self._transform,
            max_samples=None,
        )

    @property
    def max_samples(self) -> int | None:
        return self._max_samples

    def to_pandas(self) -> pd.DataFrame:
        return samples_to_pandas(self)

    def to_indexable(self) -> IndexableSplitIterator[T_Output]:
        return IndexableSplitIterator(
            base_iterator=list(self.base_iterator), transform=self._transform
        )
