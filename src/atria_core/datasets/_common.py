from __future__ import annotations

import enum
from typing import TypeVar

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.registry import ModuleConfig
from atria_core.types import BaseDataInstance


class FileStorageType(str, enum.Enum):
    MSGPACK = "msgpack"
    DELTALAKE = "deltalake"


@pydantic_dataclass(frozen=True)
class DatasetConfig(ModuleConfig):
    """Abstract base -- never registered/instantiated directly, only
    concrete dataset configs (e.g. Tobacco3482Config) are, each
    implementing its own build_module() returning its own dataset class."""

    dataset_name: str | None = None
    config_name: str = "default"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    seed: int = 42


@pydantic_dataclass(frozen=True)
class HuggingfaceDatasetConfig(DatasetConfig):
    hf_repo: str = ""
    hf_config_name: str = ""


class DatasetLoadingMode(str, enum.Enum):
    in_memory = "in_memory"
    local_streaming = "local_streaming"
    online_streaming = "online_streaming"


T_DatasetConfig = TypeVar("T_DatasetConfig", bound=DatasetConfig)
T_HuggingfaceDatasetConfig = TypeVar(
    "T_HuggingfaceDatasetConfig", bound=HuggingfaceDatasetConfig
)
T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)
