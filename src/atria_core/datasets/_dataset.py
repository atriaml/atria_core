from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Generic, TypeVar, cast, get_args, get_origin

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_DOWNLOAD_PATH,
)
from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.datasets._snapshot_store import DatasetSnapshotStore
from atria_core.datasets._split_iterators import (
    IndexableSplitIterator,
    IterableSplitIterator,
)
from atria_core.logger import get_logger
from atria_core.registry import ConfigurableModule
from atria_core.registry._module_config import ModuleConfig
from atria_core.types import DatasetMetadata, DatasetSplitType, RepresentationMixin
from atria_core.types._data_instance._base import DataInstance

logger = get_logger(__name__)


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


@pydantic_dataclass(frozen=True)
class DatasetConfig(ModuleConfig):
    """Validated params for a dataset. A config only describes -- it does not
    build anything; the dataset takes one, not the other way round."""


T_DataInstance = TypeVar("T_DataInstance", bound=DataInstance, default=DataInstance)
T_DatasetConfig = TypeVar(
    "T_DatasetConfig", bound="DatasetConfig", default=DatasetConfig
)


def _resolve_data_model(dataset_cls: type[Any]) -> type[DataInstance]:
    """Resolve the first ``Dataset`` generic argument without loading a sample."""

    def resolve(
        cls: type[Any], type_vars: dict[TypeVar, Any]
    ) -> type[DataInstance] | None:
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base) or base
            args = tuple(type_vars.get(arg, arg) for arg in get_args(base))

            if origin is Dataset and args and isinstance(args[0], type):
                return cast("type[DataInstance]", args[0])

            parameters = getattr(origin, "__parameters__", ())
            resolved = resolve(origin, dict(zip(parameters, args, strict=False)))
            if resolved is not None:
                return resolved
        return None

    return resolve(dataset_cls, {}) or DataInstance


class Dataset(
    ABC,
    RepresentationMixin,
    ConfigurableModule[T_DatasetConfig],
    Generic[T_DataInstance, T_DatasetConfig],
):
    """The config class is resolved from the second generic argument, which
    lets the dataset be constructed with no arguments at all::

        class Tobacco3482(Dataset[SinglePageDocumentInstance, Tobacco3482Config]): ...


        Tobacco3482()  # default config
        Tobacco3482(Tobacco3482Config(load_ocr=True))  # explicit config
    """

    __abstract__ = True
    __requires_access_token__ = False
    __extract_downloads__ = True
    __repr_fields__ = (
        "data_model",
        "data_dir",
        "split_iterators",
    )

    def __rich_repr__(self) -> Any:
        yield from super().__rich_repr__()
        config = asdict(self.config)
        if config:
            yield "config", config

    def __init__(
        self,
        *,
        config: T_DatasetConfig | None = None,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        dataset_dir_name: str | None = None,
    ) -> None:
        """Build every split iterator eagerly, then write a source snapshot.

        Args:
            config: Params for this dataset. Defaults to its generic config type.
            data_dir: Where to read and write data. Defaults to the shared
                cache directory named after the config or the class.
            access_token: Credential for datasets behind authentication.
            split: Build only this split, instead of every available one.
            dataset_dir_name: Dataset name override for use

        Raises:
            TypeError: If `config` is not an instance of this dataset's config class.
        """
        super().__init__(config)
        print("self.config", self.config)
        data_dir = _validate_data_dir(
            data_dir=Path(data_dir)
            if data_dir is not None
            else _DEFAULT_ATRIA_DATASETS_CACHE_DIR
            / (dataset_dir_name or type(self).__name__)
        )
        self._data_dir: Path = Path(data_dir)
        self._build_split_iterators(
            data_dir=data_dir, split=split, access_token=access_token
        )
        self._persist_snapshot()

    @property
    def data_dir(self) -> Path:
        """Directory this dataset reads from and writes its snapshot to."""
        return self._data_dir

    def _build_split_iterators(
        self,
        data_dir: str,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
    ) -> None:
        self._download(data_dir=data_dir, access_token=access_token)
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
    def train(
        self,
    ) -> IndexableSplitIterator[T_DataInstance] | IterableSplitIterator[T_DataInstance]:
        """The train split. Raises ValueError if this dataset has none."""
        return self.split_iterator(DatasetSplitType.train)

    @property
    def validation(
        self,
    ) -> IndexableSplitIterator[T_DataInstance] | IterableSplitIterator[T_DataInstance]:
        """The validation split. Raises ValueError if this dataset has none."""
        return self.split_iterator(DatasetSplitType.validation)

    @property
    def test(
        self,
    ) -> IndexableSplitIterator[T_DataInstance] | IterableSplitIterator[T_DataInstance]:
        """The test split. Raises ValueError if this dataset has none."""
        return self.split_iterator(DatasetSplitType.test)

    @property
    def metadata(self) -> DatasetMetadata:
        """Description, citation, homepage and labels for this dataset."""
        return self._metadata()

    @property
    def data_model(self) -> type[T_DataInstance]:
        """The declared type of samples produced by this dataset."""
        return cast("type[T_DataInstance]", _resolve_data_model(type(self)))

    @property
    def split_iterators(
        self,
    ) -> dict[
        DatasetSplitType,
        IndexableSplitIterator[T_DataInstance] | IterableSplitIterator[T_DataInstance],
    ]:
        """Every split this dataset built, keyed by split type."""
        return {split: self.split_iterator(split) for split in self._split_iterators}

    def split_exists(self, split: DatasetSplitType) -> bool:
        """Return whether this dataset built the given split."""
        return split in self._split_iterators

    def split_iterator(
        self, split: DatasetSplitType, max_samples: int | None = None
    ) -> IndexableSplitIterator[T_DataInstance] | IterableSplitIterator[T_DataInstance]:
        """Return one split, optionally capped at `max_samples` samples.

        Args:
            split: Which split to return.
            max_samples: Cap on samples exposed. None means no cap.

        Raises:
            ValueError: If the split does not exist, or `max_samples` is negative.
        """
        if split not in self._split_iterators:
            raise ValueError(f"Split '{split}' does not exist for this dataset.")
        split_iterator = self._split_iterators[split]
        if max_samples is None:
            return split_iterator
        if max_samples < 0:
            raise ValueError(
                f"Maximum sample count for split '{split}' cannot be negative."
            )
        return split_iterator.limit(max_samples)

    def _persist_snapshot(self) -> None:
        DatasetSnapshotStore.write_source_snapshot(dataset=self, data_dir=self.data_dir)

    @abstractmethod
    def _build_split_iterator(
        self,
        split: DatasetSplitType,
        data_dir: str,
    ) -> Sequence[Any] | Iterable[Any]:
        pass

    @abstractmethod
    def _build_input_transform(self) -> Callable[[Any], T_DataInstance]:
        pass

    @abstractmethod
    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        pass

    @abstractmethod
    def _metadata(self) -> DatasetMetadata:
        pass

    def _download_urls(self) -> list[UrlSpec]:
        """Return the files this dataset downloads. Empty means none."""
        return []

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        """Download this dataset's source files into `data_dir`.

        Args:
            data_dir: Directory the files are downloaded into.
            access_token: Substituted into URLs containing `{access_token}`.

        Returns:
            Each downloaded file's final path, keyed by its name.
        """
        from atria_core.datasets._download._download_manager import AtriaDownloadManager

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
