from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from atria_core.datasets._cached_dataset import CachedDataset, _SafeTupleLoader
from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CONFIG_PATH,
    _DEFAULT_ATRIA_DATASETS_METADATA_PATH,
    _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR,
    _DEFAULT_SNAPSHOT_PATH,
)
from atria_core.datasets._dataset_builders import (
    ComposedTransform,
    PreprocessTransform,
    _default_data_dir,
    _validate_data_dir,
    transform_hash,
)
from atria_core.logger import get_logger
from atria_core.types import BaseDataInstance, DatasetSplitType

if TYPE_CHECKING:
    from atria_core.datasets._dataset import Dataset

logger = get_logger(__name__)


class Cacher:
    """Applied from the outside to a Dataset: builds-or-reuses an on-disk
    cached snapshot and returns a CachedDataset handle for reading it back.
    Never touches the passed-in dataset's own already-built split iterators
    or user transforms -- it drives its own build via the dataset's hooks,
    using its own materialization settings (store_artifacts/resize_images/
    image_max_size), which are the only things that determine what gets
    persisted."""

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
        access_token: str | None = None,
        overwrite_existing: bool = False,
    ) -> CachedDataset[Any]:
        return self._cache(
            dataset,
            data_dir=data_dir,
            split=split,
            access_token=access_token,
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
        access_token: str | None = None,
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
            access_token=access_token,
            overwrite_existing=overwrite_existing,
            transform=transform,
        )

    def _cache(
        self,
        dataset: Dataset[Any, Any],
        *,
        data_dir: str | None,
        split: DatasetSplitType | None,
        access_token: str | None,
        overwrite_existing: bool,
        transform: Callable[[Any], Any] | None,
    ) -> CachedDataset[Any]:
        from atria_core.datasets._storage._storage_manager import StorageManager

        resolved_data_dir = _validate_data_dir(
            data_dir or _default_data_dir(type(dataset).__name__)
        )

        base_transform = PreprocessTransform(
            materialize_content=self._store_artifacts,
            resize_images=self._resize_images,
            image_max_size=self._image_max_size,
        )
        output_transform = (
            ComposedTransform([base_transform, transform])
            if transform is not None
            else base_transform
        )

        unique_path = self._compute_cache_path(
            dataset, resolved_data_dir, output_transform
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

        dataset._download(resolved_data_dir, access_token)

        for s in dataset._available_splits(resolved_data_dir):
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

            split_iterator = dataset._build_split_iterator(
                s, resolved_data_dir, output_transform=output_transform
            )
            logger.info(f"Caching split [{s.value}] to {storage_manager.storage_dir}")
            storage_manager.write_split(split_iterator=split_iterator)

        Cacher._save_dataset_info(
            str(storage_manager.storage_dir),
            storage_manager.config_name,
            dataset.config.to_dict(),
            dataset.metadata.to_dict(),
        )
        Cacher._save_snapshot(
            storage_dir=storage_manager.storage_dir,
            config_name=storage_manager.config_name,
            config_hash=dataset.config.hash,
            data_model=dataset.data_model,
            storage_type=self._storage_type,
            dataset_class_name=type(dataset).__name__,
        )
        return CachedDataset(unique_path)

    def _compute_cache_path(
        self,
        dataset: Dataset[Any, Any],
        data_dir: str,
        output_transform: Callable[[Any], Any],
    ) -> Path:
        """Single source of truth for cache uniqueness -- Cacher's job, not
        StorageManager's. Folds the dataset config hash, this dataset
        class's name, the storage backend's prefix, and a hash of whatever
        is actually being written per-sample (output_transform, which may
        include a process_and_cache transform) into the cache directory
        name, so different backends/materialization settings/process
        transforms never collide."""
        from atria_core.datasets._storage._storage_manager import StorageManager

        storage_manager_cls = StorageManager.resolve_class(self._storage_type)
        storage_dir = Path(data_dir) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
        config_name = (
            f"{storage_manager_cls.storage_prefix}/"
            f"{type(dataset).__name__}-{dataset.config.hash}"
        )
        hash_ = transform_hash(output_transform)
        if hash_ is not None:
            config_name += f"-{hash_}"
        return storage_dir / config_name

    @classmethod
    def validate_cache(cls, path: Path | str) -> bool:
        path = Path(path)
        snapshot_file = path / _DEFAULT_SNAPSHOT_PATH
        if not snapshot_file.exists():
            return False
        with open(snapshot_file) as f:
            snapshot = yaml.load(f, Loader=_SafeTupleLoader)
        required_keys = {
            "dataset_class_name",
            "data_model",
            "storage_type",
            "config_name",
            "config_hash",
        }
        return not (required_keys - snapshot.keys())

    @staticmethod
    def _write_yaml(file_path: Path, data: dict[str, Any]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    @classmethod
    def _save_dataset_info(
        cls,
        storage_dir: str,
        config_name: str,
        config: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        config_file_path = (
            Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
        )
        logger.info("Saving dataset configuration to %s", config_file_path)
        cls._write_yaml(config_file_path, config)

        metadata_file_path = (
            Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
        )
        logger.info("Saving dataset metadata to %s", metadata_file_path)
        cls._write_yaml(metadata_file_path, metadata)

    @classmethod
    def _save_snapshot(
        cls,
        storage_dir: Path | str,
        config_name: str,
        config_hash: str,
        data_model: type[BaseDataInstance],
        storage_type: FileStorageType,
        dataset_class_name: str,
    ) -> None:
        snapshot = {
            "storage_type": storage_type.value,
            "data_model": f"{data_model.__module__}.{data_model.__qualname__}",
            "dataset_class_name": dataset_class_name,
            "config_name": config_name,
            "config_hash": config_hash,
        }
        snapshot_path = Path(storage_dir) / config_name / _DEFAULT_SNAPSHOT_PATH
        logger.info("Saving dataset snapshot to %s", snapshot_path)
        cls._write_yaml(snapshot_path, snapshot)
