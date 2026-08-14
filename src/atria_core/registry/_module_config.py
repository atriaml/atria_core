from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import TYPE_CHECKING, Any, Self, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

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
    """Base class for serializable configs. Fields must be JSON-safe
    primitives, nested ModuleConfig instances, or enums.

    A config only describes params -- it never builds anything. Whatever the
    config configures takes it as a constructor argument."""

    def __post_init__(self) -> None:
        # ci/lint.sh runs mypy with --follow-imports=skip, which makes pydantic
        # itself Any, so @pydantic_dataclass does not register this class as a
        # dataclass for the checker. It is one at runtime; the cast states that.
        for field in dataclasses.fields(cast("DataclassInstance", self)):
            name = field.name
            value = getattr(self, name)
            if isinstance(value, ModuleConfig | enum.Enum):
                continue
            assert _is_json_safe(value), (
                f"{type(self).__name__}.{name} must be a JSON-safe primitive, a "
                f"nested ModuleConfig, or an enum -- got {type(value).__name__}. "
                "Lists/dicts of configs or enums aren't supported."
            )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field in dataclasses.fields(cast("DataclassInstance", self)):
            name = field.name
            value = getattr(self, name)
            if isinstance(value, ModuleConfig):
                data[name] = value.to_dict()
            elif isinstance(value, enum.Enum):
                data[name] = value.value
            else:
                data[name] = value
        data["_target_"] = f"{type(self).__module__}.{type(self).__qualname__}"
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        from hydra.utils import instantiate

        config: Self = instantiate(data)
        return config

    @property
    def hash(self) -> str:
        """Stable short hash of this config's field values -- e.g. for
        deriving a unique on-disk cache path per distinct config."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
