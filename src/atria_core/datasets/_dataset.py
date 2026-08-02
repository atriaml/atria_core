from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

from atria_core.datasets._common import T_BaseDataInstance, T_DatasetConfig
from atria_core.datasets._dataset_builders import (
    ComposedTransform,
    PreprocessTransform,
    _default_data_dir,
    _validate_data_dir,
)
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

if TYPE_CHECKING:
    from atria_core.datasets._cached_dataset import CachedDataset
    from atria_core.datasets._common import FileStorageType

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
    __repr_fields__ = {"data_model", "data_dir", "split_iterators"}

    def __init__(self, config: T_DatasetConfig) -> None:
        super().__init__(config)
        self._split_iterators: dict[
            DatasetSplitType, SplitIterator[T_BaseDataInstance]
        ] = {}
        self._data_dir: str | None = None
        self._downloaded_files: dict[str, Path] = {}

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

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata()

    @property
    def downloaded_files(self) -> dict[str, Path]:
        return self._downloaded_files

    @property
    def data_dir(self) -> str | None:
        return self._data_dir

    def load(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        enable_cached_splits: bool = False,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        num_processes: int = 8,
        cached_storage_type: FileStorageType | None = None,
        allowed_keys: set[str] | None = None,
        split_iterator_type: type[SplitIterator[T_BaseDataInstance]] = SplitIterator,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> (
        Dataset[T_DatasetConfig, T_BaseDataInstance] | CachedDataset[T_BaseDataInstance]
    ):
        """enable_cached_splits=False (default): live split iterators via
        _load_splits(). enable_cached_splits=True: delegates to cache(),
        which is self-sufficient and returns a CachedDataset."""
        from atria_core.datasets._common import FileStorageType as _FileStorageType

        if enable_cached_splits:
            return self.cache(
                data_dir=data_dir,
                split=split,
                access_token=access_token,
                cached_storage_type=cached_storage_type or _FileStorageType.DELTALAKE,
                overwrite_existing_cached=overwrite_existing_cached,
                store_artifact_content=store_artifact_content,
                max_cache_image_size=max_cache_image_size,
                num_processes=num_processes,
                allowed_keys=allowed_keys,
                split_iterator_type=split_iterator_type,
                train_transform=train_transform,
                eval_transform=eval_transform,
            )
        return self._load_splits(
            data_dir=data_dir,
            split=split,
            access_token=access_token,
            split_iterator_type=split_iterator_type,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def _load_splits(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        split_iterator_type: type[SplitIterator[T_BaseDataInstance]] = SplitIterator,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> Dataset[T_DatasetConfig, T_BaseDataInstance]:
        """Live, uncached iteration -- doesn't read from or write to disk."""
        from atria_core.datasets._dataset_builders import (
            _prepare_downloads,
            _prepare_split,
        )

        resolved = _validate_data_dir(data_dir or _default_data_dir(self))
        self._data_dir = resolved
        self._downloaded_files = _prepare_downloads(self, resolved, access_token) or {}
        split_iterators: dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]] = {}
        for s in self._available_splits():
            if split is not None and s != split:
                continue
            tf = (
                train_transform
                if s == DatasetSplitType.train
                else (eval_transform or train_transform)
            )
            split_iterators[s] = _prepare_split(
                self,
                s,
                resolved,
                split_iterator_type,
                user_transform=tf,
                for_cache=False,
            )
        self._split_iterators = split_iterators
        return self

    def cache(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        cached_storage_type: FileStorageType | None = None,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        num_processes: int = 8,
        allowed_keys: set[str] | None = None,
        split_iterator_type: type[SplitIterator[T_BaseDataInstance]] = SplitIterator,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> CachedDataset[T_BaseDataInstance]:
        """Build (or reuse) an on-disk cached snapshot of this dataset.
        Self-sufficient -- doesn't require load() to have been called
        first. train/eval_transform are applied at runtime by the returned
        CachedDataset, never baked into the cache itself."""
        from atria_core.datasets._cached_dataset import CachedDataset
        from atria_core.datasets._common import FileStorageType as _FileStorageType
        from atria_core.datasets._dataset_builders import (
            _prepare_downloads,
            _prepare_split,
        )
        from atria_core.datasets._storage._storage_manager import StorageManager

        cached_storage_type = cached_storage_type or _FileStorageType.DELTALAKE
        resolved_data_dir = _validate_data_dir(data_dir or _default_data_dir(self))
        storage_manager = StorageManager.create(
            cached_storage_type,
            data_dir=resolved_data_dir,
            num_processes=num_processes,
            dataset=self,
        )
        unique_path = storage_manager.storage_dir / storage_manager.config_name

        if (
            unique_path.exists()
            and not overwrite_existing_cached
            and CachedDataset.validate_cache(unique_path)
        ):
            logger.info(f"Loading existing cached dataset from {unique_path}")
            return CachedDataset(
                path=unique_path,
                allowed_keys=allowed_keys,
                train_transform=train_transform,
                eval_transform=eval_transform,
            ).load()

        self._data_dir = resolved_data_dir
        self._downloaded_files = (
            _prepare_downloads(self, resolved_data_dir, access_token) or {}
        )

        split_iterators: dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]] = {}
        for s in self._available_splits():
            if split is not None and s != split:
                continue
            base_iterator = (
                self._split_iterators[s].base_iterator
                if s in self._split_iterators
                else None
            )
            split_iterators[s] = _prepare_split(
                self,
                s,
                resolved_data_dir,
                split_iterator_type,
                materialize_content=store_artifact_content,
                resize_images=max_cache_image_size is not None,
                image_max_size=max_cache_image_size,
                for_cache=True,
                base_iterator=base_iterator,
            )

        for s, split_iterator in split_iterators.items():
            split_exists = storage_manager.split_exists(s)
            if split_exists and overwrite_existing_cached:
                logger.warning(f"Overwriting existing cached split {s.value}")
                storage_manager.purge_split(s)
                split_exists = False
            if not split_exists:
                logger.info(
                    f"Caching split [{s.value}] to {storage_manager.storage_dir}"
                )
                storage_manager.write_split(split_iterator=split_iterator)
            else:
                logger.info(
                    f"Skipping cached split {s.value} at {storage_manager.split_dir(s)}"
                )

        CachedDataset.save_dataset_info(
            str(storage_manager.storage_dir),
            storage_manager.config_name,
            self.config.to_dict(),
            self.metadata.to_dict(),
        )
        CachedDataset.save_snapshot(
            storage_dir=storage_manager.storage_dir,
            config_name=storage_manager.config_name,
            config_hash=self.config.hash,
            data_model=self.data_model,
            storage_type=cached_storage_type,
            dataset_name=self.config.dataset_name,
            dataset_class_name=self.__class__.__name__,
        )
        return CachedDataset(
            path=unique_path,
            allowed_keys=allowed_keys,
            train_transform=train_transform,
            eval_transform=eval_transform,
        ).load()

    def apply_transforms(
        self,
        train_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
        eval_transform: Callable[[T_BaseDataInstance], T_BaseDataInstance]
        | None = None,
    ) -> None:
        for key, split_iterator in self._split_iterators.items():
            if key == DatasetSplitType.train and train_transform is not None:
                split_iterator.output_transform = ComposedTransform(  # type: ignore[assignment]
                    [PreprocessTransform(), train_transform]
                )
            elif (
                key in {DatasetSplitType.validation, DatasetSplitType.test}
                and eval_transform is not None
            ):
                split_iterator.output_transform = ComposedTransform(  # type: ignore[assignment]
                    [PreprocessTransform(), eval_transform]
                )

    def split_exists(self, split: DatasetSplitType) -> bool:
        return split in self._split_iterators

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        return self.__data_model__

    @property
    def input_transform(
        self,
    ) -> DatasetInputTransform[T_BaseDataInstance, T_DatasetConfig]:
        return self.__input_transform__(self.data_model, self.config)

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
        raise NotImplementedError(
            "Subclasses must implement the `_custom_download` method to handle "
            "specific download logic."
        )

    @abstractmethod
    def _metadata(self) -> DatasetMetadata:
        raise NotImplementedError("Subclasses must implement the `_metadata` method.")

    @abstractmethod
    def _available_splits(self) -> list[DatasetSplitType]:
        raise NotImplementedError(
            "Subclasses must implement the `_available_splits` method."
        )

    @abstractmethod
    def _split_iterator(self, split: DatasetSplitType, data_dir: str) -> Iterable[Any]:
        raise NotImplementedError(
            "Subclasses must implement the `_split_iterator` method to provide "
            "an iterator for the specified dataset split."
        )


class ImageDataset(Dataset[T_DatasetConfig, ImageInstance], Generic[T_DatasetConfig]):
    __abstract__: bool = True
    __data_model__ = ImageInstance


class DocumentDataset(
    Dataset[T_DatasetConfig, DocumentInstance], Generic[T_DatasetConfig]
):
    __abstract__: bool = True
    __data_model__ = DocumentInstance
