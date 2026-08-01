from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._utilities._repr import RepresentationMixin


@dataclass(frozen=True, repr=False, eq=False)
class BaseDataModel(RepresentationMixin):
    """Common base for the plain @dataclass record types in atria_core.types.
    frozen=True here because subclasses must match it (a frozen dataclass
    can't inherit from a non-frozen one) -- every subclass repeats
    frozen=True in its own @dataclass(...) call, since decorator params
    aren't otherwise inherited. eq=False here too: with zero fields of its
    own, a generated BaseDataModel.__eq__ would say any two same-class
    instances are equal, silently overriding the identity-based fallback
    that eq=False subclasses (numpy-array-holding ones) rely on."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_dict()."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseDataModel:
        raise NotImplementedError(f"{cls.__name__} does not implement from_dict().")
