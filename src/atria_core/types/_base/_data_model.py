from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict

from atria_core.logger import get_logger
from atria_core.types._base._ops._table_serialization_utils import (
    _extract_pyarrow_schema,
    _flatten_dict,
    _unflatten_dict,
)
from atria_core.types._utilities._repr import RepresentationMixin

if TYPE_CHECKING:
    import pyarrow as pa

    from atria_core.types._base._ops._base_ops import StandardOps

logger = get_logger(__name__)


def _load_any(value):
    if isinstance(value, BaseDataModel):
        return value.load()
    if isinstance(value, list):
        return [_load_any(v) for v in value]
    if isinstance(value, dict):
        return {k: _load_any(v) for k, v in value.items()}
    return value


class BaseDataModel(  # type: ignore[misc]
    RepresentationMixin,
    BaseModel,
):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    # -------------------------------------
    # Bound service object
    # -------------------------------------
    @property
    def ops(self) -> StandardOps:
        from atria_core.types._base._ops._base_ops import StandardOps

        return StandardOps(self)

    def load(self) -> Self:
        loaded_fields = {}
        for name in self.__class__.model_fields:
            loaded_fields[name] = _load_any(getattr(self, name))
        new_instance = self.model_copy(update=loaded_fields)
        return new_instance

    @classmethod
    def table_schema(cls) -> dict[str, Any]:
        return _extract_pyarrow_schema(cls)

    @classmethod
    def table_schema_flattened(cls) -> dict[str, Any]:
        return _flatten_dict(cls.table_schema())

    @classmethod
    def pa_schema(cls) -> pa.Schema:
        try:
            import pyarrow as pa

            schema_items = list(cls.table_schema_flattened().items())
            return pa.schema(schema_items)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create PyArrow schema for {cls.__name__}"
            ) from e

    def to_row(
        self, include_none: bool = True, exclude: set[str] | None = None
    ) -> dict[str, Any]:
        import pyarrow as pa

        schema = self.table_schema_flattened()
        data = _flatten_dict(self.model_dump(exclude=exclude))

        row = {}
        for key in schema:
            value = data.get(key)
            if value is not None or include_none:
                row[key] = value
            if row[key] is not None:
                try:
                    pa.scalar(row[key], type=schema[key])
                except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                    raise TypeError(
                        f"Expected type {schema[key]} for field {key}, got {type(row[key])}"
                    ) from e
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        try:
            return cls(**_unflatten_dict(row, cls.table_schema()))

        except Exception as e:
            raise RuntimeError(f"Failed to create {cls.__name__} from row") from e

    def get_table_fields(self) -> dict[str, Any]:
        schema_fields = set(self.table_schema_flattened().keys())
        flattened_data = _flatten_dict(self.model_dump())
        return {k: v for k, v in flattened_data.items() if k in schema_fields}

    def update(self, **kwargs: Any) -> Self:
        return self.model_copy(update=kwargs)
