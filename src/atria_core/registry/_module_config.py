from __future__ import annotations

import enum
import hashlib
import json
import types
import typing
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from atria_core.types._json import JSON_PRIMITIVES as _JSON_PRIMITIVES
from atria_core.types._json import JSONPrimitive as JSONPrimitive
from atria_core.types._json import ParamDict as ParamDict
from atria_core.types._json import ParamList as ParamList


def _is_union(annotation: Any) -> bool:
    return typing.get_origin(annotation) in (typing.Union, types.UnionType)


def _is_json_safe_annotation(annotation: Any) -> bool:
    """Whether values of `annotation` are guaranteed to serialize to JSON.

    Mirrors `_is_json_safe`, but decides from the declared type rather than
    from a value, so a config that could never serialize is rejected when the
    class is defined instead of when it is first constructed. Enums and nested
    configs are handled by the caller: like `_is_json_safe`, this allows them
    only as a whole field, never inside a container.
    """
    if annotation in _JSON_PRIMITIVES:
        return True

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if _is_union(annotation):
        return all(_is_json_safe_annotation(arg) for arg in args)
    if origin is typing.Literal:
        return all(isinstance(value, _JSON_PRIMITIVES) for value in args)
    if origin in (list, set, frozenset, tuple):
        # tuple[X, ...] carries an Ellipsis arg, which is not a type.
        return all(_is_json_safe_annotation(arg) for arg in args if arg is not Ellipsis)
    if origin is dict:
        key, value = args
        return key is str and _is_json_safe_annotation(value)
    return False


def _is_config_field_annotation(annotation: Any) -> bool:
    """Whether `annotation` is allowed as a config field: a JSON-safe type, an
    enum, a nested config, or a union of those."""
    if isinstance(annotation, type) and issubclass(
        annotation, ModuleConfig | enum.Enum
    ):
        return True
    if _is_union(annotation):
        return all(
            _is_config_field_annotation(arg) for arg in typing.get_args(annotation)
        )
    return _is_json_safe_annotation(annotation)


class ModuleConfig(BaseModel):
    """Base class for serializable configs. Fields must be JSON-safe
    primitives, nested ModuleConfig instances, or enums.

    A config only describes params -- it never builds anything. Whatever the
    config configures takes it as a constructor argument.

    A nested config field must be annotated with the exact class it holds.
    Pydantic validates a nested value into its declared type, so annotating a
    field with a config base class and assigning a subclass drops the
    subclass's own fields on the round trip; a discriminated union is the way
    to hold more than one variant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Reject field declarations a config could never serialize.

        Checked against the annotation, so a config that cannot round-trip
        fails when its class is defined rather than when it is first
        constructed or, worse, when a cache is reopened.
        """
        for name, field in cls.model_fields.items():
            if not _is_config_field_annotation(field.annotation):
                raise TypeError(
                    f"{cls.__name__}.{name} is annotated {field.annotation!r}, "
                    "which is not JSON-safe. A config field must be a JSON "
                    "primitive, an enum, a nested ModuleConfig, or a "
                    "list/tuple/dict of those."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return this config as a JSON-safe dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a config from `to_dict()` output, validating every field.

        Raises:
            ValidationError: If a field is unknown or holds the wrong type.
        """
        return cls.model_validate(data)

    @property
    def hash(self) -> str:
        """Stable short hash of this config's field values -- e.g. for
        deriving a unique on-disk cache path per distinct config."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
