from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets._cacher import Cacher
from atria_core.datasets._common import FileStorageType
from atria_core.datasets._dataset import Dataset
from atria_core.datasets._registry import datasets
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType

logger = get_logger(__name__)


class DatasetBuilder:
    """Builds a dataset in steps: load it, then optionally cache it.

    `load()` resolves a dataset by its registered name; `cache()` and
    `process_and_cache()` write it to disk, or reuse a matching existing
    cache. Each step returns `self`, and `build()` hands back the result.
    """

    def __init__(self) -> None:
        self._dataset: Dataset[Any, Any] | None = None

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
        """Load the dataset registered under `name`.

        Args:
            name: Registered name of the dataset.
            data_dir: Where to read and write data.
            access_token: Credential for datasets behind authentication.
            split: Build only this split, instead of every available one.
            streaming: Only meaningful for a Hugging-Face-backed dataset --
                see `DatasetRegistry.create`/`HuggingfaceDataset.__init__`.
            params: Values for the dataset's config fields.
        """
        self._dataset = datasets.create(
            name,
            data_dir=data_dir,
            access_token=access_token,
            split=split,
            streaming=streaming,
            **params,
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
        """Write the loaded dataset to disk, or reuse a matching cache.

        Keyword arguments mirror `Cacher.__init__` and `Cacher.cache`, which
        remain the source of truth for their defaults.

        Raises:
            ValueError: If `load()` was not called first.
        """
        dataset = self._require_loaded_dataset(step_name="cache")
        cacher = Cacher(
            storage_type,
            num_processes=num_processes,
            use_ray=use_ray,
            store_images_to_files=store_images_to_files,
            resize_images=resize_images,
            image_max_size=image_max_size,
        )
        self._dataset = cacher.cache(
            dataset,
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
        """Like `cache()`, but writes `transform(sample)` for each sample.

        Raises:
            ValueError: If `load()` was not called first.
        """
        dataset = self._require_loaded_dataset(step_name="process_and_cache")
        cacher = Cacher(
            storage_type,
            num_processes=num_processes,
            use_ray=use_ray,
            store_images_to_files=store_images_to_files,
            resize_images=resize_images,
            image_max_size=image_max_size,
        )
        self._dataset = cacher.process_and_cache(
            dataset,
            transform,
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
            overwrite_existing=overwrite_existing,
        )
        return self

    def build(self) -> Dataset[Any, Any]:
        """Return the built dataset.

        Raises:
            ValueError: If `load()` was not called.
        """
        return self._require_loaded_dataset(step_name="build")

    def _require_loaded_dataset(self, *, step_name: str) -> Dataset[Any, Any]:
        if self._dataset is None:
            raise ValueError(f"call load() before {step_name}()")
        return self._dataset
