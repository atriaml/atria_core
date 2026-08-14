from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atria_core.datasets import DatasetRegistry, datasets
from atria_core.registry import RegistryStore
from atria_core.types import DatasetSplitType

# Registering the synthetic dataset is a side effect of importing this module.
from tests.datasets.test_dataset_end_to_end import SyntheticDataset  # noqa: F401

_SYNTHETIC_PATH = f"{SyntheticDataset.__module__}.{SyntheticDataset.__qualname__}"


def test_create_builds_the_registered_class_with_the_param_applied(
    tmp_path: Path,
) -> None:
    dataset = datasets.create("synthetic", max_train_samples=2, data_dir=str(tmp_path))

    assert isinstance(dataset, SyntheticDataset)
    assert dataset.config.max_train_samples == 2


def test_create_passes_runtime_arguments_outside_the_config(tmp_path: Path) -> None:
    dataset = datasets.create(
        "synthetic", split=DatasetSplitType.train, data_dir=str(tmp_path)
    )

    assert list(dataset.split_iterators) == [DatasetSplitType.train]
    assert "data_dir" not in dataset.config.to_dict()
    assert "split" not in dataset.config.to_dict()


def test_create_rejects_a_param_the_config_does_not_declare(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="does not accept 'nonsense'"):
        datasets.create("synthetic", nonsense=1, data_dir=str(tmp_path))


def test_create_rejects_a_param_value_the_config_rejects(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="max_train_samples"):
        datasets.create("synthetic", max_train_samples="lots", data_dir=str(tmp_path))


def test_unknown_name_raises_with_a_did_you_mean_hint() -> None:
    with pytest.raises(KeyError, match="Did you mean 'synthetic'"):
        datasets.create("synthetc")


def test_registering_a_non_dataset_is_rejected() -> None:
    registry: DatasetRegistry = DatasetRegistry("test_registry.datasets")

    with pytest.raises(TypeError, match="only accepts subclasses of Dataset"):

        @registry.register("not_a_dataset")
        class NotADataset:
            pass


def test_registering_a_duplicate_name_is_rejected() -> None:
    registry: DatasetRegistry = DatasetRegistry("test_registry.datasets")
    registry.register("synthetic")(SyntheticDataset)

    with pytest.raises(ValueError, match="already has an entry named 'synthetic'"):
        registry.register("synthetic")(SyntheticDataset)


def test_store_roundtrip_resolves_a_name_by_import_path(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    RegistryStore.dump(path, datasets.to_dict())

    restored: DatasetRegistry = DatasetRegistry("test_registry.datasets")
    restored.from_dict(RegistryStore.load(path))

    # from_dict stores the path verbatim; it is resolved on first lookup.
    assert restored.to_dict()["synthetic"] == _SYNTHETIC_PATH
    dataset = restored.create("synthetic", data_dir=str(tmp_path))
    assert isinstance(dataset, SyntheticDataset)
