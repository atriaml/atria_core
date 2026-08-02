from __future__ import annotations

import random
import sys
import traceback
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Self

from atria_core.datasets._common import T_BaseDataInstance
from atria_core.types import DatasetSplitType
from atria_core.types._utilities._repr import RepresentationMixin

if TYPE_CHECKING:
    import pandas as pd
    from rich.pretty import RichReprResult  # type: ignore[attr-defined]


class InstanceTransform(Generic[T_BaseDataInstance]):
    def __init__(
        self,
        data_model: type[T_BaseDataInstance],
        input_transform: Callable[[Any], T_BaseDataInstance] | None = None,
        output_transform: Callable[
            [T_BaseDataInstance], T_BaseDataInstance | list[T_BaseDataInstance]
        ]
        | None = None,
    ) -> None:
        self._data_model = data_model
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(
        self, index: int, input: Any
    ) -> T_BaseDataInstance | list[T_BaseDataInstance]:
        if self._input_transform is not None:
            data_instance = self._input_transform(input)
        else:
            data_instance = input

        assert isinstance(data_instance, self._data_model), (
            f"self._input_transform(sample) should return {self._data_model}, but got {type(data_instance)}"
        )

        result: T_BaseDataInstance | list[T_BaseDataInstance] = data_instance
        if self._output_transform is not None:
            result = self._output_transform(data_instance)
        return result


class SplitIterator(
    Sequence[
        T_BaseDataInstance | list[T_BaseDataInstance] | tuple[int, T_BaseDataInstance]
    ],
    RepresentationMixin,
    Generic[T_BaseDataInstance],
):
    """Abstract base: owns transform application (input_transform/
    output_transform), max_len, subset_indices, and the enable/disable_tf
    toggle storage writers use -- all concretely, not by probing an
    injected base_iterator. Concrete raw-access shapes (IndexableSplitIterator,
    IterableSplitIterator) implement how samples actually get read."""

    __repr_fields__ = {"input_transform", "output_transform", "subset_indices"}

    def __init__(
        self,
        split: DatasetSplitType,
        data_model: type[T_BaseDataInstance],
        input_transform: Callable[[Any], T_BaseDataInstance] | None = None,
        output_transform: Callable[
            [T_BaseDataInstance], T_BaseDataInstance | list[T_BaseDataInstance]
        ]
        | None = None,
        max_len: int | None = None,
        subset_indices: list[int] | None = None,
    ) -> None:
        self._split = split
        self._max_len = max_len
        self._subset_indices = subset_indices
        self._tf = InstanceTransform[T_BaseDataInstance](
            input_transform=input_transform,
            data_model=data_model,
            output_transform=output_transform,
        )
        self._tf_enabled = True

    def enable_tf(self) -> None:
        self._tf_enabled = True

    def disable_tf(self) -> None:
        self._tf_enabled = False

    @property
    def split(self) -> DatasetSplitType:
        return self._split

    @property
    def input_transform(self) -> Callable[[Any], T_BaseDataInstance] | None:
        return self._tf._input_transform

    @property
    def output_transform(
        self,
    ) -> (
        Callable[[T_BaseDataInstance], T_BaseDataInstance | list[T_BaseDataInstance]]
        | None
    ):
        return self._tf._output_transform

    @output_transform.setter
    def output_transform(
        self,
        value: Callable[
            [T_BaseDataInstance], T_BaseDataInstance | list[T_BaseDataInstance]
        ],
    ) -> None:
        self._tf._output_transform = value

    @property
    def subset_indices(self) -> list[int] | None:
        return self._subset_indices

    @subset_indices.setter
    def subset_indices(self, indices: list[int]) -> None:
        self._subset_indices = indices

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        return self._tf._data_model

    def dataframe(self) -> pd.DataFrame:
        raise RuntimeError(
            "This split iterator does not support dataframe representation."
        )

    def fetch_sample_by_id(self, sample_id: str) -> T_BaseDataInstance:
        raise RuntimeError(
            "This split iterator does not support retrieval by sample ID."
        )

    def __rich_repr__(self) -> RichReprResult:
        yield from super().__rich_repr__()
        try:
            yield "num_rows", len(self)
        except Exception:
            yield "num_rows", "unknown"


class IndexableSplitIterator(SplitIterator[T_BaseDataInstance]):
    """Concrete shape for random-access raw sources: implement _raw_getitem
    and _raw_len; __getitem__/__iter__/__len__/get_random_subset all work
    from those two, with no capability probing."""

    @abstractmethod
    def _raw_getitem(self, index: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _raw_len(self) -> int:
        raise NotImplementedError

    def _raw_getitems(self, indices: list[int]) -> list[Any]:
        return [self._raw_getitem(index) for index in indices]

    def __len__(self) -> int:
        n = (
            len(self._subset_indices)
            if self._subset_indices is not None
            else self._raw_len()
        )
        return min(self._max_len, n) if self._max_len is not None else n

    def __iter__(
        self,
    ) -> Iterator[
        T_BaseDataInstance | list[T_BaseDataInstance] | tuple[int, T_BaseDataInstance]
    ]:
        try:
            for index in range(len(self)):
                yield self[index]
        except Exception as e:
            raise RuntimeError(
                "".join(traceback.format_exception(*sys.exc_info()))
            ) from e

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> T_BaseDataInstance | list[T_BaseDataInstance] | tuple[int, T_BaseDataInstance]:
        try:
            if isinstance(index, list):
                return self.__getitems__(index)
            if self._subset_indices is not None:
                index = self._subset_indices[index]
            if self._tf_enabled:
                return self._tf(index, self._raw_getitem(index))
            return index, self._raw_getitem(index)
        except Exception as e:
            raise RuntimeError(
                "".join(traceback.format_exception(*sys.exc_info()))
            ) from e

    def __getitems__(
        self, indices: list[int]
    ) -> list[
        T_BaseDataInstance | list[T_BaseDataInstance] | tuple[int, T_BaseDataInstance]
    ]:
        try:
            if self._subset_indices is not None:
                indices = [self._subset_indices[idx] for idx in indices]
            raw_items = self._raw_getitems(indices)
            if self._tf_enabled:
                return [
                    self._tf(index, item)
                    for index, item in zip(indices, raw_items, strict=True)
                ]
            return list(zip(indices, raw_items, strict=True))
        except Exception as e:
            raise RuntimeError(
                "".join(traceback.format_exception(*sys.exc_info()))
            ) from e

    def get_random_subset(self, subset_size: int, seed: int = 42) -> Self:
        dataset_indices = list(range(len(self)))
        random.seed(seed)
        random.shuffle(dataset_indices)

        copy_split_iterator = deepcopy(self)
        copy_split_iterator.subset_indices = dataset_indices[:subset_size]
        return copy_split_iterator


class IterableSplitIterator(SplitIterator[T_BaseDataInstance]):
    """Concrete shape for stream-only raw sources (e.g. HF streaming
    datasets): implement _raw_iter; only sequential iteration is
    supported, matching what the source itself can actually do."""

    @abstractmethod
    def _raw_iter(self) -> Iterator[Any]:
        raise NotImplementedError

    def _raw_len(self) -> int | None:
        """Override if the source can report its length cheaply without
        being consumed; None means "unknown", which is fine unless
        max_len is also unset."""
        return None

    def __iter__(
        self,
    ) -> Iterator[
        T_BaseDataInstance | list[T_BaseDataInstance] | tuple[int, T_BaseDataInstance]
    ]:
        try:
            if self._subset_indices is not None:
                raise RuntimeError(
                    "You are trying to iterate over a subset of the dataset, "
                    "but this split iterator does not support indexing."
                )
            for index, sample in enumerate(self._raw_iter()):
                if self._tf_enabled:
                    yield self._tf(index, sample)
                else:
                    yield index, sample

                if self._max_len is not None and index + 1 >= self._max_len:
                    break
        except Exception as e:
            raise RuntimeError(
                "".join(traceback.format_exception(*sys.exc_info()))
            ) from e

    def __len__(self) -> int:
        n = self._raw_len()
        if n is not None:
            return min(self._max_len, n) if self._max_len is not None else n
        if self._max_len is not None:
            return self._max_len
        raise RuntimeError(
            "This split iterator does not support length calculation. Set "
            "max_len, or override _raw_len() if the source can report it."
        )

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> T_BaseDataInstance | list[T_BaseDataInstance] | tuple[int, T_BaseDataInstance]:
        raise RuntimeError(
            "This split iterator does not support indexed access; iterate it instead."
        )
