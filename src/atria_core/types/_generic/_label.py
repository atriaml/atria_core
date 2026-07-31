from atria_core.types._base._data_model import BaseDataModel
from atria_core.types._pydantic import (
    IntField,
    StrField,
)


class Label(BaseDataModel):
    value: IntField
    name: StrField
