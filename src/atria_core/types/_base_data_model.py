from __future__ import annotations

from typing import Any

from atria_core.types._utilities._repr import RepresentationMixin


class BaseDataModel(RepresentationMixin):
    """Common base for the plain @dataclass record types in atria_core.types."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_dict()."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseDataModel:
        raise NotImplementedError(f"{cls.__name__} does not implement from_dict().")
