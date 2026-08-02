from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Self

from pydantic.dataclasses import dataclass as pydantic_dataclass

_JSON_PRIMITIVES = (str, int, float, bool, type(None))


def _is_json_safe(value: Any) -> bool:
    if isinstance(value, _JSON_PRIMITIVES):
        return True
    if isinstance(value, list | tuple):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


@pydantic_dataclass(frozen=True)
class ModuleConfig:
    """Base class for registerable configs. Fields must be JSON-safe
    primitives, nested ModuleConfig instances, or enums. Subclasses
    implement build_module() to construct whatever they configure."""

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, ModuleConfig | enum.Enum):
                continue
            assert _is_json_safe(value), (
                f"{type(self).__name__}.{field.name} must be a JSON-safe primitive, a "
                f"nested ModuleConfig, or an enum -- got {type(value).__name__}. "
                "Lists/dicts of configs or enums aren't supported."
            )

    def build_module(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_module() -- every config builds something."
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, ModuleConfig):
                data[field.name] = value.to_dict()
            elif isinstance(value, enum.Enum):
                data[field.name] = value.value
            else:
                data[field.name] = value
        data["_target_"] = f"{type(self).__module__}.{type(self).__qualname__}"
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        from hydra.utils import instantiate

        return instantiate(data)  # type: ignore[no-any-return]

    @property
    def hash(self) -> str:
        """Stable short hash of this config's field values -- e.g. for
        deriving a unique on-disk cache path per distinct config."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
