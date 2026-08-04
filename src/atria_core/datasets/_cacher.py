from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Sized
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atria_core.datasets._cached_dataset import CachedDataset
from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR,
)
from atria_core.datasets._dataset import _default_data_dir, _validate_data_dir
from atria_core.datasets._snapshot import (
    PROCESSED_DATASET_STAGE,
    RAW_DATASET_STAGE,
    DatasetSnapshot,
)
from atria_core.datasets._snapshot_store import DatasetSnapshotStore
from atria_core.datasets._split_iterators import Compose
from atria_core.logger import get_logger
from atria_core.transforms.functional import image as image_functional
from atria_core.types import (
    BaseDataInstance,
    DatasetSplitType,
    Image,
    ImageInstance,
    PdfPage,
    SinglePageDocumentInstance,
)

if TYPE_CHECKING:
    from atria_core.datasets._dataset import Dataset

logger = get_logger(__name__)


def transform_hash(transform: Callable[[Any], Any] | None) -> str | None:
    """Storage layer doesn't know or care what a transform does -- it only
    needs a stable identity to fold into the cache path. Every transform
    reaching this point is already required to be picklable (workers
    receive them via Pool/Ray), so hashing the pickled bytes directly
    gives a deterministic identity for any transform, class instance or
    plain function, with no per-type special-casing."""
    if transform is None:
        return None
    return hashlib.md5(pickle.dumps(transform)).hexdigest()[:8]


class PreprocessTransform:
    """Flag-driven output transform applied per-sample before writing to
    storage. materialize_content=True makes to_dict() embed content as
    bytes instead of requiring a file path -- needed for msgpack/tar-shard
    storage, which bundles many small binary blobs into shard files rather
    than reading/writing them one file at a time."""

    def __init__(
        self,
        materialize_content: bool = True,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
    ) -> None:
        self._materialize_content = materialize_content
        self._resize_images = resize_images
        self._image_max_size = image_max_size

    def __call__(self, sample: BaseDataInstance) -> BaseDataInstance:
        if isinstance(sample, ImageInstance):
            processed = self._process_visual(sample.image)
            assert isinstance(processed, Image)
            return replace(sample, image=processed)
        if isinstance(sample, SinglePageDocumentInstance):
            return replace(sample, visual=self._process_visual(sample.visual))
        return sample

    def _process_visual(self, visual: Image | PdfPage) -> Image | PdfPage:
        if self._materialize_content:
            visual = visual.load()
        if self._resize_images and visual.content is not None:
            resized = self._resize(Image.from_source(visual.content))
            visual = replace(visual, content=resized.require_content())
        return visual

    def _resize(self, image: Image) -> Image:
        assert self._image_max_size is not None
        if isinstance(self._image_max_size, tuple):
            return image_functional.resize(
                image, width=self._image_max_size[0], height=self._image_max_size[1]
            )
        return image_functional.resize_with_aspect_ratio(
            image, max_size=self._image_max_size
        )


def _infer_data_model(dataset: Dataset[Any, Any]) -> type[Any]:
    """No __data_model__ declaration exists in the new Dataset design --
    infer the sample class from an actual sample instead."""
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
    """Applied from the outside to a Dataset: builds-or-reuses an on-disk
    cached snapshot and returns a CachedDataset handle for reading it back.
    Since Dataset.__init__ always eagerly builds every split iterator up
    front, Cacher just takes the dataset's own already-built
    dataset.split_iterators (which already have the input transform
    applied) and layers a write-time transform on top via
    .with_transform(...) -- no need to re-download or rebuild anything."""

    def __init__(
        self,
        storage_type: FileStorageType,
        *,
        num_processes: int = 8,
        use_ray: bool = False,
        store_artifacts: bool = True,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
    ) -> None:
        self._storage_type = storage_type
        self._num_processes = num_processes
        self._use_ray = use_ray
        self._store_artifacts = store_artifacts
        self._resize_images = resize_images
        self._image_max_size = image_max_size

    def cache(
        self,
        dataset: Dataset[Any, Any],
        *,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        overwrite_existing: bool = False,
    ) -> CachedDataset[Any]:
        return self._cache(
            dataset,
            data_dir=data_dir,
            split=split,
            overwrite_existing=overwrite_existing,
            transform=None,
        )

    def process_and_cache(
        self,
        dataset: Dataset[Any, Any],
        transform: Callable[[Any], Any],
        *,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        overwrite_existing: bool = False,
    ) -> CachedDataset[Any]:
        """Like cache(), but writes transform(sample) for each sample the
        dataset produces -- e.g. cache raw, then tokenize into a second
        cache. `transform` is layered on top only for this write; it never
        touches or mutates `dataset` itself."""
        return self._cache(
            dataset,
            data_dir=data_dir,
            split=split,
            overwrite_existing=overwrite_existing,
            transform=transform,
        )

    def _cache(
        self,
        dataset: Dataset[Any, Any],
        *,
        data_dir: str | None,
        split: DatasetSplitType | None,
        overwrite_existing: bool,
        transform: Callable[[Any], Any] | None,
    ) -> CachedDataset[Any]:
        from atria_core.datasets._storage._storage_manager import StorageManager

        materialize = PreprocessTransform(
            materialize_content=self._store_artifacts,
            resize_images=self._resize_images,
            image_max_size=self._image_max_size,
        )
        write_transform = (
            Compose(materialize, transform) if transform is not None else materialize
        )

        resolved_data_dir = _validate_data_dir(
            data_dir or _default_data_dir(type(dataset).__name__)
        )
        unique_path = self._compute_cache_path(
            dataset, resolved_data_dir, write_transform
        )
        storage_manager = StorageManager.create(
            self._storage_type,
            data_dir=resolved_data_dir,
            num_processes=self._num_processes,
            storage_dir=str(unique_path.parent),
            config_name=unique_path.name,
            use_ray=self._use_ray,
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

            logger.info(f"Caching split [{s.value}] to {storage_manager.storage_dir}")
            storage_manager.write_split(
                s, split_iterator.with_transform(write_transform)
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
        )
        return CachedDataset(unique_path)

    def _compute_cache_path(
        self,
        dataset: Dataset[Any, Any],
        data_dir: str,
        write_transform: Callable[[Any], Any],
    ) -> Path:
        """Single source of truth for cache uniqueness -- Cacher's job, not
        StorageManager's. Folds the dataset config hash, this dataset
        class's name, the storage backend's prefix, and a hash of whatever
        is actually being written per-sample into the cache directory
        name, so different backends/materialization settings/process
        transforms never collide."""
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
        return storage_dir / config_name

    @classmethod
    def validate_cache(cls, path: Path | str) -> bool:
        if not DatasetSnapshot.validate(path):
            return False
        return DatasetSnapshot.load(path).is_cached
