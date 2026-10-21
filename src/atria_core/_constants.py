"""Shared cache locations for Atria packages."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ATRIA_CACHE_DIR = os.environ.get(
    "DEFAULT_ATRIA_CACHE_DIR", str(Path.home() / ".cache/atria/")
)
_DEFAULT_ATRIA_MODELS_CACHE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "models/"
