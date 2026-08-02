from __future__ import annotations

import dataclasses
from collections.abc import Callable, ItemsView
from typing import TypeVar

from atria_core.registry._module_config import ModuleConfig

# Unbound to ModuleConfig on purpose: pydantic's dataclass decorator doesn't
# preserve the decorated subclass's static type across a further decorator
# (it resolves to `type[PydanticDataclass]`), so a bound TypeVar here would
# reject every `@group.register(...)` usage stacked above `@pydantic_dataclass`.
# The actual ModuleConfig constraint is enforced at runtime below.
_T = TypeVar("_T")


class RegistryGroup:
    """Pure in-memory store for one group -- register/get/list only. No
    file I/O, no branching, no knowledge of any other group; see
    RegistryStore for dump/load."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, type[ModuleConfig]] = {}

    def register(self, name: str) -> Callable[[type[_T]], type[_T]]:
        def decorator(config_cls: type[_T]) -> type[_T]:
            assert dataclasses.is_dataclass(config_cls) and issubclass(
                config_cls, ModuleConfig
            ), f"{config_cls} must be a ModuleConfig -- only configs are registerable."
            self._store[name] = config_cls
            return config_cls

        return decorator

    def get(self, name: str) -> type[ModuleConfig]:
        return self._store[name]

    def items(self) -> ItemsView[str, type[ModuleConfig]]:
        return self._store.items()
