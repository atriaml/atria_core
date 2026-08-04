from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, dataclass_transform


def _config_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _config_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_config_value(item) for item in value]
    return value


@dataclass_transform(frozen_default=True)
@dataclass(frozen=True)
class BaseTransform(ABC):
    """Dataclass base for configurable, serializable transforms."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls, frozen=True)

    @abstractmethod
    def __call__(self, value: Any) -> Any:
        pass

    def dump(self) -> dict[str, Any]:
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "params": _config_value(asdict(self)),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.dump()

    @property
    def hash(self) -> str:
        encoded = json.dumps(
            self.dump(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:8]
