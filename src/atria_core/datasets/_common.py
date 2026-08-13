from __future__ import annotations

import enum

from atria_core.logger import get_logger

logger = get_logger(__name__)


class FileStorageType(enum.StrEnum):
    """On-disk format a cached dataset is written in."""

    MSGPACK = "msgpack"
    DELTALAKE = "deltalake"


class DatasetLoadingMode(enum.StrEnum):
    """How samples are pulled: fully materialized, from local files, or streamed."""

    in_memory = "in_memory"
    local_streaming = "local_streaming"
    online_streaming = "online_streaming"
