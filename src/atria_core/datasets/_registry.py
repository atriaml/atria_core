from __future__ import annotations

from typing import Any, cast

from atria_core.datasets._dataset import Dataset
from atria_core.registry._registry import ConfigRegistry
from atria_core.types import DatasetSplitType


class DatasetRegistry(ConfigRegistry["Dataset[Any, Any]"]):
    """Maps names to Dataset classes, building them from config params.

    Code that knows which dataset it wants should import the class and
    construct it directly, which keeps the exact type. This exists for a name
    arriving as data -- a command line flag, a config file -- where the class
    cannot be chosen statically.
    """

    def __init__(self, name: str) -> None:
        # Dataset is abstract, which is the point -- it is the subclass check
        # every entry must pass, never a class this registry instantiates.
        super().__init__(name, target_type=cast("type[Dataset[Any, Any]]", Dataset))

    def create(
        self,
        name: str,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        **params: Any,
    ) -> Dataset[Any, Any]:
        """Build the dataset registered under `name`.

        Args:
            name: Registered name of the dataset.
            data_dir: Where to read and write data.
            access_token: Credential for datasets behind authentication.
            split: Build only this split, instead of every available one.
            params: Values for the dataset's config fields.

        Raises:
            KeyError: If nothing is registered under `name`.
            TypeError: If `params` holds names the dataset's config does not
                accept.
            ValidationError: If a param holds a value the config rejects.
        """
        dataset_cls = self._get_class(name)
        self._check_params(dataset_cls, params=params, name=name)
        return dataset_cls(
            config=dataset_cls.config_class()(**params),
            data_dir=data_dir,
            access_token=access_token,
            split=split,
        )

    def _check_params(
        self, target: type[Dataset[Any, Any]], *, params: dict[str, Any], name: str
    ) -> None:
        """Reject params the dataset's config class does not declare.

        The params configure the dataset, not the dataset class itself, so the
        accepted names come from its config's fields.
        """
        accepted = set(target.config_class().model_fields)
        unknown = sorted(set(params) - accepted)
        if unknown:
            raise TypeError(
                f"{name!r} ({target.__name__}) does not accept "
                f"{', '.join(repr(field) for field in unknown)}. "
                f"Accepted parameters: {', '.join(sorted(accepted)) or '<none>'}"
            )


datasets: DatasetRegistry = DatasetRegistry("datasets")
