from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, ItemsView
from typing import Generic, TypeVar

T = TypeVar("T")


def import_submodules(package: str) -> None:
    """Import every submodule under `package` recursively, so any
    Registry.register() decorators inside them run and prefill whatever
    registries they target -- lets a downstream package do one bulk import
    to populate a registry (e.g. before dumping it), instead of every
    caller needing to import each dataset/model/etc. module by hand."""
    module = importlib.import_module(package)
    if not hasattr(module, "__path__"):
        return
    for _, name, _ in pkgutil.walk_packages(
        module.__path__, prefix=f"{module.__name__}."
    ):
        importlib.import_module(name)


class Registry(Generic[T]):
    """A single named, typed name->object map. Entries are stored as
    import-path strings, never as live objects -- register() derives the
    path from the target, and get()/items() import the owning module lazily,
    only when something actually asks for the object."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, str] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(target: T) -> T:
            self._store[name] = f"{target.__module__}.{target.__qualname__}"
            return target

        return decorator

    def get(self, name: str) -> T:
        module_name, attr = self._store[name].rsplit(".", 1)
        return getattr(importlib.import_module(module_name), attr)

    def items(self) -> ItemsView[str, T]:
        return {name: self.get(name) for name in self._store}.items()

    def list(self) -> list[str]:
        return list(self._store.keys())

    def to_dict(self) -> dict[str, str]:
        return dict(self._store)

    def from_dict(self, data: dict[str, str]) -> None:
        self._store.update(data)
