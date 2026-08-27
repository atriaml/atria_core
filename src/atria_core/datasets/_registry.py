from __future__ import annotations

import inspect
from typing import Any, cast

from atria_core.datasets._dataset import Dataset, DatasetConfig
from atria_core.registry._module_config import ModuleConfig
from atria_core.registry._registry import ModuleRegistry
from atria_core.types import DatasetSplitType
from atria_core.types._data_instance._base import DataInstance


class DatasetRegistry(ModuleRegistry[Dataset[DataInstance, DatasetConfig]]):
    __registry_name__ = "datasets"

    def create(
        self,
        name: str,
        config: ModuleConfig | None = None,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        streaming: bool | None = None,
        **params: Any,
    ) -> Dataset[DataInstance, DatasetConfig]:
        """Build the dataset registered under `name`.

        Args:
            name: Registered name of the dataset.
            config: The dataset's config, built directly. Mutually exclusive
                with `params`.
            data_dir: Where to read and write data.
            access_token: Credential for datasets behind authentication.
            split: Build only this split, instead of every available one.
            streaming: Only meaningful for `HuggingfaceDataset` subclasses --
                whether to stream rows instead of materializing the dataset
                to disk first (see `HuggingfaceDataset.__init__`). Left
                unset (`None`) to use that class's own default; ignored
                entirely for a dataset that doesn't accept it.
            params: Values for the dataset's config fields. Mutually
                exclusive with `config`.

        Raises:
            KeyError: If nothing is registered under `name`.
            TypeError: If both `config` and `params` are given.
            ValidationError: If `params` holds a name the dataset's config
                does not declare, or a value it rejects.
        """
        dataset_cls = self.get(name)
        if config is None:
            config = dataset_cls.config_type()(**params)
        elif params:
            raise TypeError("cannot pass both 'config' and individual config params")
        extra_kwargs: dict[str, Any] = {}
        if streaming is not None and "streaming" in inspect.signature(dataset_cls).parameters:
            extra_kwargs["streaming"] = streaming
        return dataset_cls(
            config=cast("DatasetConfig", config),
            data_dir=data_dir,
            access_token=access_token,
            split=split,
            dataset_dir_name=name,
            **extra_kwargs,
        )


datasets = DatasetRegistry()
