from __future__ import annotations

import enum
from typing import TypeVar

from atria_core.logger import get_logger
from atria_core.registry._module_config import ModuleConfig
from atria_core.types._data_instance._base import BaseDataInstance

logger = get_logger(__name__)


T_DatasetConfig = TypeVar("T_DatasetConfig", bound=ModuleConfig)
T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)


class FileStorageType(str, enum.Enum):
    MSGPACK = "msgpack"
    DELTALAKE = "deltalake"


class DatasetLoadingMode(str, enum.Enum):
    in_memory = "in_memory"
    local_streaming = "local_streaming"
    online_streaming = "online_streaming"
