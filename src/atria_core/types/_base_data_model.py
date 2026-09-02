from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atria_core.types._utilities._repr import RepresentationMixin


@dataclass(frozen=True, repr=False, eq=False)
class BaseDataModel(RepresentationMixin):
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_dict()."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseDataModel:
        raise NotImplementedError(f"{cls.__name__} does not implement from_dict().")
