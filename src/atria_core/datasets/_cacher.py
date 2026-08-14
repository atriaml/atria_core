from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable, Sized
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from atria_core.datasets._cached_dataset import CachedDataset
from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
from atria_core.datasets._dataset import _validate_data_dir
from atria_core.datasets._snapshot import (
    PROCESSED_DATASET_STAGE,
    RAW_DATASET_STAGE,
    DatasetSnapshot,
)
from atria_core.datasets._snapshot_store import DatasetSnapshotStore
from atria_core.datasets._split_iterators import Compose
from atria_core.logger import get_logger
from atria_core.transforms import BaseTransform
from atria_core.transforms._base import _config_value
from atria_core.transforms.functional import image as image_functional
from atria_core.types import (
    DataInstance,
    DatasetSplitType,
    Image,
    ImageInstance,
    PdfPage,
    SinglePageDocumentInstance,
)

if TYPE_CHECKING:
    from atria_core.datasets._dataset import Dataset

logger = get_logger(__name__)

T_Sample = TypeVar("T_Sample", bound=DataInstance)
"""Sample type a dataset yields -- preserved through cache() into the handle."""

T_ProcessedSample = TypeVar("T_ProcessedSample", bound=DataInstance)
"""Sample type a write-time transform produces in process_and_cache()."""


def transform_hash(transform: Callable[[Any], Any] | None) -> str | None:
    """Return a stable identity for a transform or composed pipeline."""
    if transform is None:
        return None
    if isinstance(transform, BaseTransform):
        return transform.hash
    if isinstance(transform, Compose):
        components = [
            item.dump()
            if isinstance(item, BaseTransform)
            else {"pickle_hash": hashlib.sha256(pickle.dumps(item)).hexdigest()}
            for item in transform.transforms
        ]
        encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:8]
    return hashlib.md5(pickle.dumps(transform)).hexdigest()[:8]


class PreprocessTransform(BaseTransform):
    """Write-time image preparation applied to every sample before storage."""

    resize_images: bool = False
    image_max_size: int | tuple[int, int] | None = None

    def __call__(self, sample: DataInstance) -> DataInstance:
        if isinstance(sample, ImageInstance):
            processed = self._process_visual(sample.image)
            assert isinstance(processed, Image)
            return replace(sample, image=processed)
        if isinstance(sample, SinglePageDocumentInstance):
            return replace(sample, visual=self._process_visual(sample.visual))
        return sample

    def _process_visual(self, visual: Image | PdfPage) -> Image | PdfPage:
        if not isinstance(visual, Image):
            return visual

        if self.resize_images:
            visual = visual.load()
            resized = self._resize(Image.from_source(visual.require_content()))
            visual = replace(visual, content=resized.require_content())

        return visual

    def _resize(self, image: Image) -> Image:
        assert self.image_max_size is not None
        if isinstance(self.image_max_size, tuple):
            return image_functional.resize(
                image, width=self.image_max_size[0], height=self.image_max_size[1]
            )
        return image_functional.resize_with_aspect_ratio(
            image, max_size=self.image_max_size
        )


def transform_configs(
    transform: Callable[[Any], Any] | None,
) -> list[dict[str, Any]]:
    """Return snapshot metadata for configurable transforms in a pipeline."""
    if transform is None:
        return []
    transforms = transform.transforms if isinstance(transform, Compose) else [transform]
    return [
        item.dump()
        if isinstance(item, BaseTransform)
        else {
            "type": f"{type(item).__module__}.{type(item).__qualname__}",
            "params": _config_value(asdict(cast(Any, item))),
        }
        for item in transforms
        if not isinstance(item, type) and is_dataclass(item)
    ]


def _infer_data_model(dataset: Dataset[T_Sample, Any]) -> type[T_Sample]:
    """Return the sample class a dataset produces, read off its first sample.

    Raises:
        ValueError: If every split is empty, leaving no sample to inspect.
    """
    for split_iterator in dataset.split_iterators.values():
        for sample in split_iterator:
            return type(sample)
    raise ValueError(
        f"Cannot infer data_model for {type(dataset).__name__}: every split is empty."
    )


def _stored_split_counts(storage_manager: Any) -> dict[str, int]:
    counts = {}
    for dataset_split in storage_manager.get_splits():
        stored = storage_manager.read_split(dataset_split)
        counts[dataset_split.value] = (
            len(stored) if isinstance(stored, Sized) else sum(1 for _ in stored)
        )
    return counts


class Cacher:
    """Writes a dataset's splits to disk and returns a handle for reading
    them back, reusing an existing cache when one matches.

    Caching reads from the dataset's already-built split iterators and layers
    any write-time transform on top of them, so nothing is re-downloaded or
    rebuilt. The cache location is derived from the dataset's config hash, the
    storage backend, and the write transform, so distinct inputs never
    collide."""

    def __init__(
        self,
        storage_type: FileStorageType,
        *,
        num_processes: int = 8,
        use_ray: bool = False,
        store_images_to_files: bool = False,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
    ) -> None:
        self._storage_type = storage_type
        self._num_processes = num_processes
        self._use_ray = use_ray
        self._store_images_to_files = store_images_to_files
        self._resize_images = resize_images
        self._image_max_size = image_max_size

    def cache(
        self,
        dataset: Dataset[T_Sample, Any],
        *,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        max_samples: int | None = None,
        overwrite_existing: bool = False,
    ) -> CachedDataset[T_Sample]:
        """Write this dataset to disk, or reuse a matching existing cache.

        Args:
            dataset: Dataset whose already-built splits are written.
            data_dir: Root to cache under. Defaults to the dataset's own.
            split: Cache only this split, instead of every one.
            max_samples: Cap on samples written per split.
            overwrite_existing: Rewrite splits that are already cached.

        Returns:
            A handle for reading the cache back, yielding the same sample type
            the dataset does.
        """
        return self._cache(
            dataset,
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
            overwrite_existing=overwrite_existing,
            transform=None,
        )

    def process_and_cache(
        self,
        dataset: Dataset[T_Sample, Any],
        transform: Callable[[T_Sample], T_ProcessedSample],
        *,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        max_samples: int | None = None,
        overwrite_existing: bool = False,
    ) -> CachedDataset[T_ProcessedSample]:
        """Like cache(), but writes transform(sample) for each sample the
        dataset produces -- e.g. cache raw, then tokenize into a second cache.

        The returned handle yields the transform's output type, not the
        dataset's. `transform` is layered on top only for this write; it never
        touches or mutates `dataset` itself.

        Args:
            dataset: Dataset whose already-built splits are written.
            transform: Applied to each sample at write time.
            data_dir: Root to cache under. Defaults to the dataset's own.
            split: Cache only this split, instead of every one.
            max_samples: Cap on samples written per split.
            overwrite_existing: Rewrite splits that are already cached.

        Returns:
            A handle for reading the transformed cache back.
        """
        return self._cache(
            dataset,
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
            overwrite_existing=overwrite_existing,
            transform=transform,
        )

    def _cache(
        self,
        dataset: Dataset[T_Sample, Any],
        *,
        data_dir: str | None,
        split: DatasetSplitType | None,
        max_samples: int | None,
        overwrite_existing: bool,
        transform: Callable[[T_Sample], T_ProcessedSample] | None,
    ) -> CachedDataset[T_ProcessedSample]:
        from atria_core.datasets._storage._storage_manager import StorageManager

        preprocess = PreprocessTransform(
            resize_images=self._resize_images,
            image_max_size=self._image_max_size,
        )

        resolved_data_dir = _validate_data_dir(data_dir or dataset.data_dir)
        write_transform = (
            Compose(preprocess, transform) if transform is not None else preprocess
        )
        unique_path = self._compute_cache_path(
            dataset=dataset,
            data_dir=resolved_data_dir,
            write_transform=write_transform,
            max_samples=max_samples,
        )
        storage_manager = StorageManager.create(
            self._storage_type,
            data_dir=resolved_data_dir,
            num_processes=self._num_processes,
            storage_dir=str(unique_path.parent),
            config_name=unique_path.name,
            use_ray=self._use_ray,
            store_images_to_files=self._store_images_to_files,
        )

        if (
            unique_path.exists()
            and not overwrite_existing
            and Cacher.validate_cache(unique_path)
        ):
            logger.info(f"Loading existing cached dataset from {unique_path}")
            return CachedDataset(unique_path)

        data_model = _infer_data_model(dataset)

        for s, split_iterator in dataset.split_iterators.items():
            if split is not None and s != split:
                continue

            split_exists = storage_manager.split_exists(s)
            if split_exists and overwrite_existing:
                logger.warning(f"Overwriting existing cached split {s.value}")
                storage_manager.purge_split(s)
                split_exists = False
            if split_exists:
                logger.info(
                    f"Skipping cached split {s.value} at {storage_manager.split_dir(s)}"
                )
                continue

            if max_samples is not None:
                split_iterator = split_iterator.limit(max_samples)
            logger.info(f"Caching split [{s.value}] to {storage_manager.storage_dir}")
            storage_manager.write_split(
                s, split_iterator=split_iterator.with_transform(write_transform)
            )

        DatasetSnapshotStore.write_cached_snapshot(
            dataset=dataset,
            snapshot_dir=unique_path,
            config_name=storage_manager.config_name,
            config_hash=dataset.config.hash,
            data_model=data_model,
            storage_type=self._storage_type,
            dataset_stage=RAW_DATASET_STAGE
            if transform is None
            else PROCESSED_DATASET_STAGE,
            splits=_stored_split_counts(storage_manager),
            transforms=transform_configs(write_transform),
        )
        return CachedDataset(unique_path)

    def _compute_cache_path(
        self,
        dataset: Dataset[Any, Any],
        data_dir: str,
        write_transform: Callable[[Any], Any],
        max_samples: int | None = None,
    ) -> Path:
        """Single source of truth for cache uniqueness -- Cacher's job, not
        StorageManager's. Folds the dataset config hash, this dataset
        class's name, the storage backend's prefix, a hash of whatever is
        actually being written per-sample, and any sample cap into the cache
        directory name, so different backends/materialization settings/process
        transforms/sample counts never collide."""
        from atria_core.datasets._storage._storage_manager import StorageManager

        storage_manager_cls = StorageManager.resolve_class(self._storage_type)
        storage_dir = Path(data_dir) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
        config_name = (
            f"{storage_manager_cls.storage_prefix}/"
            f"{type(dataset).__name__}-{dataset.config.hash}"
        )
        hash_ = transform_hash(write_transform)
        if hash_ is not None:
            config_name += f"-{hash_}"
        if self._store_images_to_files:
            config_name += "-images-files"
        if max_samples is not None:
            # A capped cache holds fewer samples than the dataset it came
            # from, so it must never be reused as though it were complete.
            config_name += f"-max{max_samples}"
        return storage_dir / config_name

    @classmethod
    def validate_cache(cls, path: Path | str) -> bool:
        """Return whether `path` holds a complete cached-dataset snapshot."""
        if not DatasetSnapshot.validate(path):
            return False
        return DatasetSnapshot.load(path).is_cached
