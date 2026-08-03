from __future__ import annotations

import enum

from atria_core.logger import get_logger

logger = get_logger(__name__)


class FileStorageType(str, enum.Enum):
    MSGPACK = "msgpack"
    DELTALAKE = "deltalake"


class DatasetLoadingMode(str, enum.Enum):
    in_memory = "in_memory"
    local_streaming = "local_streaming"
    online_streaming = "online_streaming"
