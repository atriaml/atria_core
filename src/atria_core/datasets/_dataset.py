from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_DOWNLOAD_PATH,
)
from atria_core.datasets._split_iterators import (
    IndexableSplitIterator,
    IterableSplitIterator,
)
from atria_core.logger import get_logger
from atria_core.registry import ConfigurableModule
from atria_core.registry._module_config import ModuleConfig
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
)
from atria_core.types._data_instance._base import BaseDataInstance

logger = get_logger(__name__)


T_DatasetConfig = TypeVar("T_DatasetConfig", bound=ModuleConfig)
T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)


def _validate_data_dir(data_dir: str | Path) -> str:
    data_dir = Path(data_dir)
    if data_dir.exists():
        assert data_dir.is_dir(), (
            f"Data directory `{data_dir.absolute()}` exists but is not a directory."
        )
    else:
        logger.warning(
            f"Data directory `{data_dir.absolute()}` does not exist. Creating it."
        )
        data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def _default_data_dir(class_name: str) -> str:
    return str(_DEFAULT_ATRIA_DATASETS_CACHE_DIR / class_name)


@pydantic_dataclass(frozen=True)
class DatasetConfig(ModuleConfig):
    pass


class Dataset(
    ConfigurableModule[T_DatasetConfig],
    Generic[T_DatasetConfig, T_BaseDataInstance],
    ABC,
):
    __abstract__ = True
    __requires_access_token__ = False
    __extract_downloads__ = True

    def __init__(
        self,
        config: T_DatasetConfig,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
    ) -> None:
        super().__init__(config)
        data_dir = _validate_data_dir(
            data_dir or _default_data_dir(type(self).__name__)
        )
        self._build_split_iterators(data_dir, split=split, access_token=access_token)

    def _build_split_iterators(
        self,
        data_dir: str,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
    ) -> None:
        self._download(data_dir, access_token)
        input_transform = self._build_input_transform()
        self._split_iterators = {}
        for dataset_split in self._available_splits(data_dir):
            if split is not None and dataset_split != split:
                continue

            split_iterator = self._build_split_iterator(
                split=dataset_split,
                data_dir=data_dir,
            )

            if isinstance(split_iterator, Sequence):
                split_iterator = IndexableSplitIterator(
                    base_iterator=split_iterator,
                    transform=input_transform,
                )

            elif isinstance(split_iterator, Iterable):
                split_iterator = IterableSplitIterator(
                    base_iterator=split_iterator,
                    transform=input_transform,
                )
            else:
                raise TypeError(
                    f"Unsupported split iterator type: {type(split_iterator).__name__}"
                )

            self._split_iterators[dataset_split] = split_iterator

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata()

    @property
    def split_iterators(
        self,
    ) -> dict[
        DatasetSplitType,
        IndexableSplitIterator[T_BaseDataInstance]
        | IterableSplitIterator[T_BaseDataInstance],
    ]:
        return self._split_iterators

    def split_exists(self, split: DatasetSplitType) -> bool:
        return split in self._split_iterators

    def split_iterator(
        self, split: DatasetSplitType
    ) -> (
        IndexableSplitIterator[T_BaseDataInstance]
        | IterableSplitIterator[T_BaseDataInstance]
    ):
        if split not in self._split_iterators:
            raise ValueError(f"Split '{split}' does not exist for this dataset.")
        return self._split_iterators[split]

    @abstractmethod
    def _build_split_iterator(
        self,
        split: DatasetSplitType,
        data_dir: str,
    ) -> Sequence[Any] | Iterable[Any]:
        pass

    @abstractmethod
    def _build_input_transform(
        self, **kwargs: Any
    ) -> Callable[[Any], T_BaseDataInstance]:
        pass

    @abstractmethod
    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        pass

    @abstractmethod
    def _metadata(self) -> DatasetMetadata:
        pass

    def _download_urls(self) -> dict[str, str] | list[str]:
        return []

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        """Default: run the generic Downloader against _download_urls().
        Subclasses with nonstandard fetch logic (e.g. HuggingfaceDataset)
        override this outright -- no override-detection, just polymorphism."""
        from atria_core.datasets._download._download_manager import (
            AtriaDownloadManager,
        )

        if self.__requires_access_token__ and access_token is None:
            logger.warning(
                "access_token must be passed to download this dataset. "
                f"See `{self.metadata.homepage}` for instructions to get the access token"
            )

        download_urls = self._download_urls()
        if not download_urls:
            return {}
        download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
        download_dir.mkdir(parents=True, exist_ok=True)
        download_manager = AtriaDownloadManager(
            data_dir=Path(data_dir), download_dir=download_dir
        )
        downloaded = download_manager.download_and_extract(
            download_urls,
            extract=self.__extract_downloads__,
            access_token=access_token,
        )
        logger.info(f"Downloaded files {downloaded}")
        return downloaded
