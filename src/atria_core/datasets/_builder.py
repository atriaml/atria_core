from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atria_core.datasets._cacher import Cacher
from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR
from atria_core.datasets._dataset import Dataset
from atria_core.datasets._registry import datasets
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType

logger = get_logger(__name__)


@dataclass(frozen=True)
class _LoadRequest:
    """Params queued by `DatasetBuilder.load()`, applied on `build()`."""

    name: str
    data_dir: str | None
    access_token: str | None
    split: DatasetSplitType | None
    streaming: bool | None
    params: dict[str, Any]


@dataclass(frozen=True)
class _CacheRequest:
    """Params queued by `DatasetBuilder.cache()`/`process_and_cache()`,
    applied on `build()`."""

    storage_type: FileStorageType
    transform: Callable[[Any], Any] | None
    num_processes: int
    use_ray: bool
    store_images_to_files: bool
    resize_images: bool
    image_max_size: int | tuple[int, int] | None
    data_dir: str | None
    split: DatasetSplitType | None
    max_samples: int | None
    overwrite_existing: bool


class DatasetBuilder:
    """Declares how to build a dataset, then builds it in one shot.

    `load()` and `cache()`/`process_and_cache()` only record what was asked
    for -- nothing runs until `build()`. When both a load and a cache step
    were queued, `build()` first checks whether a matching cache already
    exists using only the dataset's class name and config hash, neither of
    which requires the dataset to be constructed or its data downloaded. If a
    cache is found, it's reused directly and the dataset is never built. If
    not, `build()` constructs and builds the dataset, then writes the cache.
    """

    def __init__(self) -> None:
        self._load_request: _LoadRequest | None = None
        self._cache_request: _CacheRequest | None = None

    def load(
        self,
        name: str,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        streaming: bool | None = None,
        **params: Any,
    ) -> DatasetBuilder:
        """Queue loading the dataset registered under `name`.

        Args:
            name: Registered name of the dataset.
            data_dir: Where to read and write data.
            access_token: Credential for datasets behind authentication.
            split: Build only this split, instead of every available one.
            streaming: Only meaningful for a Hugging-Face-backed dataset --
                see `DatasetRegistry.load`/`HuggingfaceDataset.__init__`.
            params: Values for the dataset's config fields.
        """
        self._load_request = _LoadRequest(
            name=name,
            data_dir=data_dir,
            access_token=access_token,
            split=split,
            streaming=streaming,
            params=params,
        )
        return self

    def cache(
        self,
        storage_type: FileStorageType,
        *,
        num_processes: int = 8,
        use_ray: bool = False,
        store_images_to_files: bool = False,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        max_samples: int | None = None,
        overwrite_existing: bool = False,
    ) -> DatasetBuilder:
        """Queue writing the loaded dataset to disk, or reusing a matching cache.

        Keyword arguments mirror `Cacher.__init__` and `Cacher.cache`, which
        remain the source of truth for their defaults.
        """
        self._cache_request = _CacheRequest(
            storage_type=storage_type,
            transform=None,
            num_processes=num_processes,
            use_ray=use_ray,
            store_images_to_files=store_images_to_files,
            resize_images=resize_images,
            image_max_size=image_max_size,
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
            overwrite_existing=overwrite_existing,
        )
        return self

    def process_and_cache(
        self,
        storage_type: FileStorageType,
        transform: Callable[[Any], Any],
        *,
        num_processes: int = 8,
        use_ray: bool = False,
        store_images_to_files: bool = False,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        max_samples: int | None = None,
        overwrite_existing: bool = False,
    ) -> DatasetBuilder:
        """Like `cache()`, but queues writing `transform(sample)` for each sample."""
        self._cache_request = _CacheRequest(
            storage_type=storage_type,
            transform=transform,
            num_processes=num_processes,
            use_ray=use_ray,
            store_images_to_files=store_images_to_files,
            resize_images=resize_images,
            image_max_size=image_max_size,
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
            overwrite_existing=overwrite_existing,
        )
        return self

    def build(self) -> Dataset[Any, Any]:
        """Run the queued steps and return the resulting dataset.

        If both `load()` and `cache()`/`process_and_cache()` were queued,
        an existing matching cache is reused without ever constructing or
        downloading the source dataset. Otherwise the source dataset is
        constructed (which downloads and builds its split iterators), then
        cached if a cache step was queued.

        Raises:
            ValueError: If `load()` was not called.
        """
        if self._load_request is None:
            raise ValueError("call load() before build()")

        if self._cache_request is not None:
            cached = self._build_from_cache_if_present()
            if cached is not None:
                return cached

        dataset = self._build_loaded_dataset()
        if self._cache_request is None:
            return dataset
        return self._build_cache(dataset)

    def _build_from_cache_if_present(self) -> Dataset[Any, Any] | None:
        from atria_core.datasets._cached_dataset import CachedDataset

        assert self._load_request is not None
        assert self._cache_request is not None
        load_request = self._load_request
        cache_request = self._cache_request

        if cache_request.overwrite_existing:
            return None

        dataset_cls, config = datasets.resolve_config(
            load_request.name, **load_request.params
        )
        data_dir = (
            cache_request.data_dir
            or load_request.data_dir
            or str(_DEFAULT_ATRIA_DATASETS_CACHE_DIR / load_request.name)
        )
        cacher = Cacher(
            cache_request.storage_type,
            num_processes=cache_request.num_processes,
            use_ray=cache_request.use_ray,
            store_images_to_files=cache_request.store_images_to_files,
            resize_images=cache_request.resize_images,
            image_max_size=cache_request.image_max_size,
        )
        existing_path = cacher.find_existing_cache_path(
            dataset_cls.__name__,
            config.hash,
            data_dir=data_dir,
            transform=cache_request.transform,
            max_samples=cache_request.max_samples,
        )
        if existing_path is None:
            return None
        logger.info(f"Loading existing cached dataset from {existing_path}")
        return CachedDataset(existing_path)

    def _build_loaded_dataset(self) -> Dataset[Any, Any]:
        assert self._load_request is not None
        load_request = self._load_request
        return datasets.create(
            load_request.name,
            data_dir=load_request.data_dir,
            access_token=load_request.access_token,
            split=load_request.split,
            streaming=load_request.streaming,
            **load_request.params,
        )

    def _build_cache(self, dataset: Dataset[Any, Any]) -> Dataset[Any, Any]:
        assert self._cache_request is not None
        cache_request = self._cache_request
        cacher = Cacher(
            cache_request.storage_type,
            num_processes=cache_request.num_processes,
            use_ray=cache_request.use_ray,
            store_images_to_files=cache_request.store_images_to_files,
            resize_images=cache_request.resize_images,
            image_max_size=cache_request.image_max_size,
        )
        if cache_request.transform is None:
            return cacher.cache(
                dataset,
                data_dir=cache_request.data_dir,
                split=cache_request.split,
                max_samples=cache_request.max_samples,
                overwrite_existing=cache_request.overwrite_existing,
            )
        return cacher.process_and_cache(
            dataset,
            cache_request.transform,
            data_dir=cache_request.data_dir,
            split=cache_request.split,
            max_samples=cache_request.max_samples,
            overwrite_existing=cache_request.overwrite_existing,
        )
