from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import yaml

from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CONFIG_PATH,
    _DEFAULT_SNAPSHOT_PATH,
)


@dataclasses.dataclass(frozen=True)
class DatasetSnapshot:
    """Portable description and completion marker for a cached dataset.

    Fields added after the original snapshot format are optional so snapshots
    produced by older atria_core releases remain readable.
    """

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    storage_type: FileStorageType
    data_model: str
    dataset_class_name: str
    config_name: str
    config_hash: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    created_at: str | None = None
    splits: dict[str, int] = dataclasses.field(default_factory=dict)
    path: Path | None = dataclasses.field(default=None, compare=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        storage_type: FileStorageType,
        data_model: str,
        dataset_class_name: str,
        config_name: str,
        config_hash: str,
        splits: dict[str, int],
    ) -> DatasetSnapshot:
        return cls(
            storage_type=storage_type,
            data_model=data_model,
            dataset_class_name=dataset_class_name,
            config_name=config_name,
            config_hash=config_hash,
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
            storage_type=FileStorageType(data["storage_type"]),
            data_model=str(data["data_model"]),
            dataset_class_name=str(data["dataset_class_name"]),
            config_name=str(data["config_name"]),
            config_hash=str(data["config_hash"]),
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
        # CachedDataset has always treated metadata.yaml as optional, so it
        # must not become a discovery requirement for legacy snapshots.
        return (directory / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH).is_file()

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
