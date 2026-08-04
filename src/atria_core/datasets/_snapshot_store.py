from __future__ import annotations

from collections.abc import Mapping, Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from atria_core.datasets._constants import _DEFAULT_SNAPSHOT_PATH
from atria_core.datasets._snapshot import (
    CACHED_DATASET_SNAPSHOT_KIND,
    RAW_DATASET_STAGE,
    SOURCE_DATASET_SNAPSHOT_KIND,
    DatasetSnapshot,
)
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType

if TYPE_CHECKING:
    from atria_core.datasets._common import FileStorageType
    from atria_core.datasets._dataset import Dataset

logger = get_logger(__name__)


class DatasetSnapshotStore:
    @staticmethod
    def _write_yaml(file_path: Path, data: dict[str, Any]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    @staticmethod
    def _split_counts(
        split_iterators: Mapping[DatasetSplitType, object],
    ) -> dict[str, int]:
        counts = {}
        for split, split_iterator in split_iterators.items():
            if isinstance(split_iterator, Sized):
                counts[split.value] = len(split_iterator)
        return counts

    @classmethod
    def write_source_snapshot(
        cls,
        dataset: Dataset[Any, Any],
        data_dir: Path | str,
    ) -> DatasetSnapshot:
        snapshot = DatasetSnapshot.create(
            storage_type=None,
            data_model=None,
            dataset_class_name=type(dataset).__name__,
            config_name=Path(data_dir).name,
            config_hash=dataset.config.hash,
            snapshot_kind=SOURCE_DATASET_SNAPSHOT_KIND,
            config=dataset.config.to_dict(),
            metadata=dataset.metadata.to_dict(),
            dataset_stage=RAW_DATASET_STAGE,
            splits=cls._split_counts(dataset.split_iterators),
            transforms=[],
        )
        cls.write_snapshot(Path(data_dir), snapshot)
        return snapshot

    @classmethod
    def write_cached_snapshot(
        cls,
        *,
        dataset: Dataset[Any, Any],
        snapshot_dir: Path | str,
        storage_type: FileStorageType,
        data_model: type[Any],
        config_name: str,
        config_hash: str,
        dataset_stage: str,
        splits: dict[str, int],
        transforms: list[dict[str, Any]] | None = None,
    ) -> DatasetSnapshot:
        snapshot = DatasetSnapshot.create(
            storage_type=storage_type,
            data_model=f"{data_model.__module__}.{data_model.__qualname__}",
            dataset_class_name=type(dataset).__name__,
            config_name=config_name,
            config_hash=config_hash,
            snapshot_kind=CACHED_DATASET_SNAPSHOT_KIND,
            config=dataset.config.to_dict(),
            metadata=dataset.metadata.to_dict(),
            dataset_stage=dataset_stage,
            splits=splits,
            transforms=transforms or [],
        )
        cls.write_snapshot(Path(snapshot_dir), snapshot)
        return snapshot

    @classmethod
    def write_snapshot(cls, directory: Path, snapshot: DatasetSnapshot) -> None:
        snapshot_path = directory / _DEFAULT_SNAPSHOT_PATH
        logger.info("Saving dataset snapshot to %s", snapshot_path)
        temporary_path = snapshot_path.with_suffix(f"{snapshot_path.suffix}.tmp")
        cls._write_yaml(temporary_path, snapshot.to_dict())
        temporary_path.replace(snapshot_path)

    @classmethod
    def discover(
        cls,
        base_dir: Path | str,
        *,
        snapshot_kind: str | None = None,
        dataset_stage: str | None = None,
    ) -> list[DatasetSnapshot]:
        snapshots = DatasetSnapshot.discover(base_dir)
        if snapshot_kind is not None:
            snapshots = [
                item for item in snapshots if item.snapshot_kind == snapshot_kind
            ]
        if dataset_stage is not None:
            snapshots = [
                item for item in snapshots if item.dataset_stage == dataset_stage
            ]
        return snapshots
