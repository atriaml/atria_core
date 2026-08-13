from __future__ import annotations

from atria_core.datasets._dataset import DatasetConfig
from atria_core.registry import ConfigRegistry


class DatasetConfigsRegistry(ConfigRegistry[DatasetConfig]):
    def __init__(self, name: str) -> None:
        super().__init__(name, target_type=DatasetConfig)

    def create(self, name: str, **params: Any) -> DatasetConfig:
        return super().create(name=name, dataset_dir_name=name, **params)


dataset_configs: DatasetConfigsRegistry = DatasetConfigsRegistry("datasets")
