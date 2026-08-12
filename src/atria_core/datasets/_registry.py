from __future__ import annotations

from atria_core.datasets._dataset import DatasetConfig
from atria_core.registry import Registry

datasets: Registry[type[DatasetConfig]] = Registry("datasets")
