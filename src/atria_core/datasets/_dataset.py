from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic

from atria_core.datasets._common import T_BaseDataInstance, T_DatasetConfig
from atria_core.datasets._constants import _DEFAULT_DOWNLOAD_PATH
from atria_core.datasets._dataset_builders import _default_data_dir, _validate_data_dir
from atria_core.datasets._exceptions import SplitNotFoundError
from atria_core.datasets._split_iterators import SplitIterator
from atria_core.logger import get_logger
from atria_core.registry import ConfigurableModule
from atria_core.types import (
    BaseDataInstance,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
)
from atria_core.types._utilities._repr import RepresentationMixin

logger = get_logger(__name__)


class DatasetInputTransform(Generic[T_BaseDataInstance, T_DatasetConfig]):
    def __init__(
        self, data_model: type[T_BaseDataInstance], config: T_DatasetConfig
    ) -> None:
        self.data_model = data_model
        self.config = config

    def __call__(self, *args: Any, **kwargs: Any) -> T_BaseDataInstance:
        assert len(args) == 1, "Expected a single positional argument 'sample'."
        assert len(kwargs) == 0, "No keyword arguments expected."
        sample = args[0]
        if isinstance(sample, self.data_model):
            return sample
        if isinstance(sample, dict):
            return self.data_model(**sample)
        raise TypeError(
            f"Cannot convert sample of type {type(sample)} to data model {self.data_model}"
        )


class Dataset(
    ConfigurableModule[T_DatasetConfig],
    RepresentationMixin,
    Generic[T_DatasetConfig, T_BaseDataInstance],
):
    __abstract__ = True
    __requires_access_token__ = False
    __extract_downloads__ = True
    __data_model__: type[T_BaseDataInstance]
    __input_transform__: type[DatasetInputTransform[Any, Any]] = DatasetInputTransform
    __split_iterator__: type[SplitIterator[Any]] | None = None
    __repr_fields__ = {"data_model", "split_iterators"}

    def __init__(
        self,
        config: T_DatasetConfig,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> None:
        """Building is Dataset's whole job: this constructor resolves the
        data dir, downloads, and builds every split iterator in one shot --
        there is no separate load()/cache() mutating self afterward.
        data_dir/downloaded_files are local variables used only to build
        _split_iterators, never stored as attributes."""
        super().__init__(config)
        self._split_iterators: dict[
            DatasetSplitType, SplitIterator[T_BaseDataInstance]
        ] = {}
        resolved = _validate_data_dir(
            data_dir or _default_data_dir(type(self).__name__)
        )
        self._custom_download(resolved, access_token)
        for s in self._available_splits(resolved):
            if split is not None and s != split:
                continue
            self._split_iterators[s] = self._build_split_iterator(
                s,
                resolved,
                output_transform=(
                    train_transform
                    if s == DatasetSplitType.train
                    else (eval_transform or train_transform)
                ),
            )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.__dict__.get("__abstract__", False):
            return

        data_model = cls.__data_model__
        if data_model is None:
            raise TypeError(
                f"Class '{cls.__name__}' must define a __data_model__ attribute "
                "to specify the type of data instances."
            )
        if not issubclass(data_model, BaseDataInstance):
            raise TypeError(
                f"Class '{cls.__name__}.__data_model__' must be a type, "
                f"got {type(data_model).__name__}: {data_model}"
            )
        assert isinstance(cls.__requires_access_token__, bool), (
            f"Class '{cls.__name__}' must define __requires_access_token__ as a boolean."
        )
        assert isinstance(cls.__extract_downloads__, bool), (
            f"Class '{cls.__name__}' must define __extract_downloads__ as a boolean."
        )

    def _build_split_iterator(
        self,
        split: DatasetSplitType,
        data_dir: str,
        output_transform: Callable[
            [T_BaseDataInstance], T_BaseDataInstance | list[T_BaseDataInstance]
        ]
        | None = None,
    ) -> SplitIterator[T_BaseDataInstance]:
        """Default: construct __split_iterator_cls__(split=split,
        data_dir=data_dir, ...) directly -- the split iterator class the
        user defines for their dataset (mirroring __input_transform__),
        not a generic wrapper around a separately-defined raw iterator.
        Subclasses whose raw source needs different construction args
        (e.g. HuggingfaceDataset, which needs its HF DatasetBuilder state)
        override this method directly instead of setting
        __split_iterator_cls__."""
        if self.__split_iterator__ is None:
            raise NotImplementedError(
                f"Class '{type(self).__name__}' must either set "
                "__split_iterator_cls__ or override _build_split_iterator()."
            )
        limits = {
            DatasetSplitType.train: self.config.max_train_samples,
            DatasetSplitType.validation: self.config.max_validation_samples,
            DatasetSplitType.test: self.config.max_test_samples,
        }
        return self.__split_iterator__(  # type: ignore[call-arg]
            split=split,
            data_dir=data_dir,
            data_model=self.data_model,
            input_transform=self.input_transform,
            output_transform=output_transform,
            max_len=limits[split],
        )

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata()

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        return self.__data_model__

    @property
    def input_transform(
        self,
    ) -> DatasetInputTransform[T_BaseDataInstance, T_DatasetConfig]:
        return self.__input_transform__(self.data_model, self.config)

    def apply_transforms(
        self,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> None:
        for key, split_iterator in self._split_iterators.items():
            if key == DatasetSplitType.train and train_transform is not None:
                split_iterator.output_transform = train_transform
            elif (
                key in {DatasetSplitType.validation, DatasetSplitType.test}
                and eval_transform is not None
            ):
                split_iterator.output_transform = eval_transform

    def split_exists(self, split: DatasetSplitType) -> bool:
        return split in self._split_iterators

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.train not in self._split_iterators:
            raise SplitNotFoundError("Training split iterator is not available.")
        return self._split_iterators[DatasetSplitType.train]

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self._split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.validation not in self._split_iterators:
            raise SplitNotFoundError("Validation split iterator is not available.")
        return self._split_iterators[DatasetSplitType.validation]

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self._split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance]:
        if DatasetSplitType.test not in self._split_iterators:
            raise SplitNotFoundError("Test split iterator is not available.")
        return self._split_iterators[DatasetSplitType.test]

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        self._split_iterators[DatasetSplitType.test] = value

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        return self._split_iterators

    def _download_urls(self) -> dict[str, str] | list[str]:
        return []

    def _custom_download(
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

    @abstractmethod
    def _metadata(self) -> DatasetMetadata:
        raise NotImplementedError("Subclasses must implement the `_metadata` method.")

    @abstractmethod
    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        raise NotImplementedError(
            "Subclasses must implement the `_available_splits` method."
        )


class ImageDataset(Dataset[T_DatasetConfig, ImageInstance], Generic[T_DatasetConfig]):
    __abstract__: bool = True
    __data_model__ = ImageInstance


class DocumentDataset(
    Dataset[T_DatasetConfig, DocumentInstance], Generic[T_DatasetConfig]
):
    __abstract__: bool = True
    __data_model__ = DocumentInstance
