from __future__ import annotations

from typing import Any, cast

from atria_core.datasets._dataset import Dataset
from atria_core.registry._registry import ConfigRegistry
from atria_core.types import DatasetSplitType


class DatasetRegistry(ConfigRegistry["Dataset"]):
    """Maps names to Dataset classes, building them from config params.

    Code that knows which dataset it wants should import the class and
    construct it directly, which keeps the exact type. This exists for a name
    arriving as data -- a command line flag, a config file -- where the class
    cannot be chosen statically.
    """

    def __init__(self, name: str) -> None:
        # Dataset is abstract, which is the point -- it is the subclass check
        # every entry must pass, never a class this registry instantiates.
        super().__init__(name, target_type=cast("type[Dataset]", Dataset))

    def create(
        self,
        name: str,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        **params: Any,
    ) -> Dataset:
        """Build the dataset registered under `name`.

        Args:
            name: Registered name of the dataset.
            data_dir: Where to read and write data.
            access_token: Credential for datasets behind authentication.
            split: Build only this split, instead of every available one.
            params: Values for the dataset's config fields.

        Raises:
            KeyError: If nothing is registered under `name`.
            ValidationError: If `params` holds a name the dataset's config
                does not declare, or a value it rejects.
        """
        dataset_cls = self._get_class(name)
        return dataset_cls(
            config=dataset_cls.config_class()(**params),
            data_dir=data_dir,
            access_token=access_token,
            split=split,
            dataset_dir_name=name,
        )


datasets: DatasetRegistry = DatasetRegistry("datasets")
