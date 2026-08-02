from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
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
    __repr_fields__ = {
        "base_iterator",
        "input_transform",
        "output_transform",
        "subset_indices",
    }

    def __init__(
        self,
        split: DatasetSplitType,
        base_iterator: Sequence[Any] | Generator[Any, None, None],
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
        self._base_iterator = base_iterator
        self._max_len = max_len
        self._subset_indices = subset_indices
        self._tf = InstanceTransform[T_BaseDataInstance](
            input_transform=input_transform,
            data_model=data_model,
            output_transform=output_transform,
        )
        self._tf_enabled = True
        self._is_iterable = isinstance(self._base_iterator, Iterable)
        self._supports_indexing = hasattr(
            self._base_iterator, "__getitem__"
        ) and hasattr(self._base_iterator, "__len__")
        self._supports_multi_indexing = hasattr(
            self._base_iterator, "__getitems__"
        ) and hasattr(self._base_iterator, "__len__")
        if not self._is_iterable:
            assert hasattr(self._base_iterator, "__len__"), (
                f"The base iterator {self._base_iterator} must implement __len__ to support indexing."
            )
            assert self._supports_indexing or self._supports_multi_indexing, (
                f"The base iterator {self._base_iterator} must implement either __iter__, __getitem__ or __getitems__"
            )

    def enable_tf(self) -> None:
        self._tf_enabled = True

    def disable_tf(self) -> None:
        self._tf_enabled = False

    @property
    def split(self) -> DatasetSplitType:
        return self._split

    @property
    def base_iterator(self) -> Iterable[Any]:
        return self._base_iterator

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
        if hasattr(self._base_iterator, "dataframe"):
            return self._base_iterator.dataframe()
        raise RuntimeError(
            "This dataset is not backed by a DataFrame or does not support dataframe representation."
        )

    def fetch_sample_by_id(self, sample_id: str) -> T_BaseDataInstance:
        if hasattr(self._base_iterator, "fetch_sample_by_id"):
            index, sample = self._base_iterator.fetch_sample_by_id(sample_id)
            if self._tf_enabled:
                return self._tf(index, sample)  # type: ignore[return-value]
            return sample  # type: ignore[no-any-return]
        raise RuntimeError("The base iterator does not support retrieval by sample ID.")

    def __iter__(
        self,
    ) -> Iterator[
        T_BaseDataInstance | list[T_BaseDataInstance] | tuple[int, T_BaseDataInstance]
    ]:
        try:
            if not self._supports_indexing:
                if self._subset_indices is not None:
                    raise RuntimeError(
                        "You are trying to iterate over a subset of the dataset, "
                        "but the base iterator does not support indexing."
                    )

                for index, sample in enumerate(self._base_iterator):
                    if self._tf_enabled:
                        yield self._tf(index, sample)
                    else:
                        yield index, sample

                    if self._max_len is not None and index + 1 >= self._max_len:
                        break
            else:
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
            assert self._supports_indexing, (
                "The base iterator does not support multi-indexing. "
                "Please use __getitem__ for single index access."
            )
            if self._subset_indices is not None:
                index = self._subset_indices[index]
            if self._tf_enabled:
                return self._tf(index, self._base_iterator[index])  # type: ignore[index]
            return index, self._base_iterator[index]  # type: ignore[index]
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
            if hasattr(self._base_iterator, "__getitems__"):
                data_instances = self._base_iterator.__getitems__(indices)
            else:
                assert self._supports_indexing, (
                    "The base iterator does not support multi-indexing. "
                    "Please use __getitem__ for single index access."
                )
                data_instances = [
                    self._base_iterator[index]  # type: ignore[index]
                    for index in indices
                ]
            if self._tf_enabled:
                return [
                    self._tf(index, data_instance)
                    for index, data_instance in zip(
                        indices, data_instances, strict=True
                    )
                ]
            return [
                (index, data_instance)
                for index, data_instance in zip(indices, data_instances, strict=True)
            ]
        except Exception as e:
            raise RuntimeError(
                "".join(traceback.format_exception(*sys.exc_info()))
            ) from e

    def __len__(self) -> int:
        if hasattr(self._base_iterator, "__len__"):
            iterator = (
                self._subset_indices
                if self._subset_indices is not None
                else self._base_iterator
            )
            if self._max_len is not None:
                return min(self._max_len, len(iterator))  # type: ignore[arg-type]
            return len(iterator)  # type: ignore[arg-type]
        elif self._max_len is not None:
            return self._max_len
        raise RuntimeError(
            "The dataset does not support length calculation. "
            "Please implement the `__len__` method in your dataset class."
        )

    def __rich_repr__(self) -> RichReprResult:
        yield from super().__rich_repr__()
        try:
            yield "num_rows", len(self)
        except Exception:
            yield "num_rows", "unknown"

    def get_random_subset(self, subset_size: int, seed: int = 42) -> Self:
        import random

        assert self._supports_indexing, (
            "The base iterator must support indexing to get a random subset."
        )

        dataset_indices = list(range(len(self)))

        random.seed(seed)
        random.shuffle(dataset_indices)

        copy_split_iterator = deepcopy(self)
        copy_split_iterator.subset_indices = dataset_indices[:subset_size]
        return copy_split_iterator


class HFSplitIterator(SplitIterator[T_BaseDataInstance]):
    def __init__(
        self,
        split: DatasetSplitType,
        base_iterator: Sequence[Any] | Generator[Any, None, None],
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
        self._base_iterator = base_iterator
        self._max_len = max_len
        self._subset_indices = subset_indices
        self._tf = InstanceTransform(
            input_transform=input_transform,
            data_model=data_model,
            output_transform=output_transform,
        )
        self._tf_enabled = True
        self._is_iterable = isinstance(self._base_iterator, Iterable)
        self._supports_indexing = False
        self._supports_multi_indexing = False
