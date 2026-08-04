from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import yaml

from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import _DEFAULT_SNAPSHOT_PATH
from atria_core.types import DatasetMetadata


RAW_DATASET_STAGE = "raw"
PROCESSED_DATASET_STAGE = "processed"
SOURCE_DATASET_SNAPSHOT_KIND = "source"
CACHED_DATASET_SNAPSHOT_KIND = "cached"


@dataclasses.dataclass(frozen=True)
class DatasetSnapshot:
    """Portable description and completion marker for a cached dataset.

    Fields added after the original snapshot format are optional so snapshots
    produced by older atria_core releases remain readable.
    """

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    storage_type: FileStorageType | None
    data_model: str | None
    dataset_class_name: str
    config_name: str | None
    config_hash: str | None
    snapshot_kind: str = CACHED_DATASET_SNAPSHOT_KIND
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    dataset_stage: str = RAW_DATASET_STAGE
    schema_version: int = CURRENT_SCHEMA_VERSION
    created_at: str | None = None
    splits: dict[str, int] = dataclasses.field(default_factory=dict)
    path: Path | None = dataclasses.field(default=None, compare=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        storage_type: FileStorageType | None,
        data_model: str | None,
        dataset_class_name: str,
        config_name: str | None,
        config_hash: str | None,
        snapshot_kind: str,
        config: dict[str, Any],
        metadata: dict[str, Any],
        dataset_stage: str,
        splits: dict[str, int],
    ) -> DatasetSnapshot:
        return cls(
            storage_type=storage_type,
            data_model=data_model,
            dataset_class_name=dataset_class_name,
            config_name=config_name,
            config_hash=config_hash,
            snapshot_kind=snapshot_kind,
            config=config,
            metadata=metadata,
            dataset_stage=dataset_stage,
            created_at=datetime.now(UTC).isoformat(),
            splits=splits,
        )

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, path: Path | None = None
    ) -> DatasetSnapshot:
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            created_at=data.get("created_at"),
            storage_type=(
                FileStorageType(data["storage_type"])
                if data.get("storage_type") is not None
                else None
            ),
            data_model=(
                str(data["data_model"]) if data.get("data_model") is not None else None
            ),
            dataset_class_name=str(data["dataset_class_name"]),
            config_name=(
                str(data["config_name"])
                if data.get("config_name") is not None
                else None
            ),
            config_hash=(
                str(data["config_hash"])
                if data.get("config_hash") is not None
                else None
            ),
            snapshot_kind=str(
                data.get("snapshot_kind", CACHED_DATASET_SNAPSHOT_KIND)
            ),
            config=dict(data.get("config") or {}),
            metadata=dict(data.get("metadata") or {}),
            dataset_stage=str(data.get("dataset_stage", RAW_DATASET_STAGE)),
            splits={
                str(name): int(count) for name, count in data.get("splits", {}).items()
            },
            path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "storage_type": self.storage_type.value,
            "data_model": self.data_model,
            "dataset_class_name": self.dataset_class_name,
            "config_name": self.config_name,
            "config_hash": self.config_hash,
            "snapshot_kind": self.snapshot_kind,
            "config": self.config,
            "metadata": self.metadata,
            "dataset_stage": self.dataset_stage,
            "splits": self.splits,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def load(cls, path: Path | str) -> DatasetSnapshot:
        directory = Path(path)
        snapshot_path = (
            directory
            if directory.name == _DEFAULT_SNAPSHOT_PATH
            else directory / _DEFAULT_SNAPSHOT_PATH
        )
        with snapshot_path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Snapshot is empty or invalid: {snapshot_path}")
        return cls.from_dict(data, path=snapshot_path.parent)

    @classmethod
    def validate(cls, path: Path | str) -> bool:
        directory = Path(path)
        if directory.name == _DEFAULT_SNAPSHOT_PATH:
            directory = directory.parent
        try:
            snapshot = cls.load(directory)
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
            return False
        if snapshot.schema_version > cls.CURRENT_SCHEMA_VERSION:
            return False
        if snapshot.snapshot_kind == SOURCE_DATASET_SNAPSHOT_KIND:
            return bool(snapshot.config)
        if (
            snapshot.storage_type is None
            or snapshot.data_model is None
            or snapshot.config_name is None
            or snapshot.config_hash is None
        ):
            return False
        if snapshot.config:
            return True
        # Legacy caches stored config in a sidecar file. New caches embed it
        # directly in snapshot.yaml and don't need any extra files to be
        # discoverable.
        return (directory / "config.yaml").is_file()

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata.from_dict(self.metadata)

    @property
    def is_cached(self) -> bool:
        return self.snapshot_kind == CACHED_DATASET_SNAPSHOT_KIND

    @classmethod
    def discover(cls, base_dir: Path | str) -> list[DatasetSnapshot]:
        base = Path(base_dir).expanduser()
        if not base.is_dir():
            return []
        snapshots = []
        for path in sorted(base.rglob(_DEFAULT_SNAPSHOT_PATH)):
            if cls.validate(path):
                snapshots.append(cls.load(path))
        return snapshots
