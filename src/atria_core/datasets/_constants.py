from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ATRIA_CACHE_DIR = os.environ.get(
    "DEFAULT_ATRIA_CACHE_DIR", str(Path.home() / ".cache/atria/")
)

_DEFAULT_ATRIA_DATASETS_CACHE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "datasets/"
_DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR = "storage"
_DEFAULT_ATRIA_DATASETS_CONFIG_PATH = "config.yaml"
_DEFAULT_ATRIA_DATASETS_METADATA_PATH = "metadata.yaml"
_DEFAULT_DOWNLOAD_PATH = ".download_cache"
_DEFAULT_SNAPSHOT_PATH = "snapshot.yaml"
