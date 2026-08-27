from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from atria_core.datasets._storage._deltalake._deltalake_row_writer import ParquetSchema
from atria_core.types._base_data_model import BaseDataModel

if TYPE_CHECKING:
    import pandas as pd


def _sample_to_row(sample: BaseDataModel) -> dict[str, Any]:
    return ParquetSchema.flatten(sample.to_dict())


def samples_to_pandas(samples: Iterable[object]) -> pd.DataFrame:
    """Convert an iterable of `BaseDataModel` samples into a flat DataFrame.

    Each sample is serialized with `to_dict()` and flattened into dotted
    column names, the same mapping used to write samples to Delta Lake, so
    a sample produces identical column names in either output.

    Raises:
        TypeError: If any sample is not a `BaseDataModel` instance.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, BaseDataModel):
            raise TypeError(
                "to_pandas() requires every sample to be a BaseDataModel instance, "
                f"got {type(sample).__name__}."
            )
        rows.append(_sample_to_row(sample))
    return pd.DataFrame(rows)
