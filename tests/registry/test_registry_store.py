from __future__ import annotations

import importlib
from pathlib import Path

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.registry import ModuleConfig, Registry, RegistryStore

_datasets: Registry[type[MockDatasetConfig]] = Registry("test_registry_store.datasets")


@_datasets.register("mock")
@pydantic_dataclass(frozen=True)
class MockDatasetConfig(ModuleConfig):
    path: str = "/data"

    def build_module(self) -> str:
        return self.path


def test_registry_register_and_get() -> None:
    assert _datasets.get("mock") is MockDatasetConfig
    assert dict(_datasets.items()) == {"mock": MockDatasetConfig}


def my_tool(text: str) -> int:
    return len(text)


def test_register_accepts_plain_callables_not_just_module_configs() -> None:
    # Registry places no runtime restriction on what's registered -- callers
    # pin what a given registry holds via its type parameter at declaration
    # time (e.g. Registry[Callable[..., Any]]), not a runtime assertion here.
    group: Registry[object] = Registry("some_group")

    group.register("my_tool")(my_tool)

    assert group.get("my_tool") is my_tool
    assert group.list() == ["my_tool"]


def test_registry_to_dict_includes_registered_config_import_paths() -> None:
    data = _datasets.to_dict()

    assert data["mock"] == (
        f"{MockDatasetConfig.__module__}.{MockDatasetConfig.__qualname__}"
    )


def test_registry_store_dump_and_load_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "registry.json"

    RegistryStore.dump(out, _datasets.to_dict())
    data = RegistryStore.load(out)
    import_path = data["mock"]

    assert (
        import_path
        == f"{MockDatasetConfig.__module__}.{MockDatasetConfig.__qualname__}"
    )

    module_name, class_name = import_path.rsplit(".", 1)
    resolved_cls = getattr(importlib.import_module(module_name), class_name)

    assert resolved_cls is MockDatasetConfig
    assert resolved_cls() == MockDatasetConfig()
