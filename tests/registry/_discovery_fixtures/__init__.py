from __future__ import annotations

from atria_core.registry import Module, ModuleConfig, ModuleRegistry


class ItemConfig(ModuleConfig):
    pass


class Item(Module[ItemConfig]):
    __abstract__ = True


class ItemRegistry(ModuleRegistry[Item]):
    __registry_name__ = "tests.registry._discovery_fixtures.items"


items = ItemRegistry()
