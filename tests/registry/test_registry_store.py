from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.registry import ModuleConfig, Registry, RegistryGroup, RegistryStore

_datasets = Registry.group("test_registry_store.datasets")


@_datasets.register("mock")
@pydantic_dataclass(frozen=True)
class MockDatasetConfig(ModuleConfig):
    path: str = "/data"

    def build_module(self) -> str:
        return self.path


def test_registry_group_register_and_get() -> None:
    assert _datasets.get("mock") is MockDatasetConfig
    assert dict(_datasets.items()) == {"mock": MockDatasetConfig}


def test_register_rejects_non_module_config() -> None:
    group = RegistryGroup("some_group")

    with pytest.raises(AssertionError):
        group.register("not_a_config")(object)


def test_registry_to_dict_includes_registered_config_import_paths() -> None:
    data = Registry.to_dict()

    assert data["test_registry_store.datasets"]["mock"] == (
        f"{MockDatasetConfig.__module__}.{MockDatasetConfig.__qualname__}"
    )


def test_registry_store_dump_and_load_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "registry.json"

    RegistryStore.dump(out, Registry.to_dict())
    data = RegistryStore.load(out)
    import_path = data["test_registry_store.datasets"]["mock"]

    assert (
        import_path
        == f"{MockDatasetConfig.__module__}.{MockDatasetConfig.__qualname__}"
    )

    module_name, class_name = import_path.rsplit(".", 1)
    resolved_cls = getattr(importlib.import_module(module_name), class_name)

    assert resolved_cls is MockDatasetConfig
    assert resolved_cls() == MockDatasetConfig()
