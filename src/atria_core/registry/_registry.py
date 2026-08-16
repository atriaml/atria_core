from __future__ import annotations

import difflib
import importlib
import pkgutil
from collections.abc import ItemsView
from typing import Any, ClassVar, cast, get_args, get_origin

from atria_core.registry._module import Module
from atria_core.registry._module_config import ModuleConfig


def import_submodules(package: str) -> None:
    module = importlib.import_module(package)
    if not hasattr(module, "__path__"):
        return
    for _, name, _ in pkgutil.walk_packages(
        module.__path__, prefix=f"{module.__name__}."
    ):
        importlib.import_module(name)


class ModuleRegistry[T_Module: Module[ModuleConfig]]:
    """Maps names to `Module` subclasses, so one can be created by name.

    `__registry_name__` and `__module_type__` are read from each subclass's
    own namespace -- `__registry_name__` for its use in error messages,
    `__module_type__` for the subclass check every registered entry must
    pass -- rather than exposed as constructor arguments.
    """

    __registry_name__: ClassVar[str]
    __module_type__: ClassVar[type[Module[ModuleConfig]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__registry_name__" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define a '__registry_name__' class variable."
            )
        for base in cls.__dict__.get("__orig_bases__", ()):
            if get_origin(base) is ModuleRegistry:
                (arg,) = get_args(base)
                module_cls = get_origin(arg) or arg
                if not (isinstance(module_cls, type) and issubclass(module_cls, Module)):
                    raise TypeError(
                        f"{cls.__name__} must specialize ModuleRegistry with a "
                        f"concrete Module subclass, got {arg!r}"
                    )
                cls.__module_type__ = module_cls
                return

    def __init__(self) -> None:
        if not hasattr(self, "__module_type__"):
            raise TypeError(
                f"{type(self).__name__} must specialize ModuleRegistry, e.g. "
                f"class MyRegistry(ModuleRegistry[MyModule]): ..."
            )
        self._store: dict[str, type[T_Module] | str] = {}

    @property
    def name(self) -> str:
        """This registry group's name, used in error messages."""
        return type(self).__registry_name__

    @property
    def module_type(self) -> type[T_Module]:
        """Class every entry in this registry must subclass."""
        return cast("type[T_Module]", type(self).__module_type__)

    def register(self, cls: type[T_Module]) -> type[T_Module]:
        name = cls.__module_name__
        if name in self._store:
            raise ValueError(f"{self.name!r} already has {name!r}")
        self._store[name] = self._validate(cls, name)
        return cls

    def get(self, name: str) -> type[T_Module]:
        entry = self._store.get(name)
        if entry is None:
            close = difflib.get_close_matches(name, self._store, n=3)
            hint = f" Did you mean {', '.join(map(repr, close))}?" if close else ""
            raise KeyError(f"{self.name!r} has no entry {name!r}.{hint}")
        if isinstance(entry, str):
            resolved = self._validate(self._import(entry, name), name)
            self._store[name] = resolved
            return resolved
        return entry

    def create(self, name: str, config: ModuleConfig | None = None) -> T_Module:
        cls = self.get(name)
        return cls(config)

    def list(self) -> list[str]:
        return list(self._store)

    def items(self) -> ItemsView[str, type[T_Module]]:
        return {name: self.get(name) for name in self._store}.items()

    def to_dict(self) -> dict[str, str]:
        return {
            name: entry
            if isinstance(entry, str)
            else f"{entry.__module__}.{entry.__qualname__}"
            for name, entry in self._store.items()
        }

    def from_dict(self, data: dict[str, str]) -> None:
        for name, path in data.items():
            if name in self._store:
                raise ValueError(f"{self.name!r} already has {name!r}")
            self._store[name] = path

    def _validate(self, target: Any, name: str) -> type[T_Module]:
        if not isinstance(target, type) or not issubclass(target, self.module_type):
            raise TypeError(
                f"{self.name!r} only accepts {self.module_type.__name__} "
                f"subclasses, got {target!r}"
            )
        return target

    def _import(self, path: str, name: str) -> Any:
        module_path, _, attr = path.rpartition(".")
        if not module_path:
            raise ValueError(f"Invalid import path {path!r} for {name!r}")
        mod = importlib.import_module(module_path)
        try:
            return getattr(mod, attr)
        except AttributeError:
            raise AttributeError(
                f"{module_path!r} has no attribute {attr!r} for entry {name!r}"
            ) from None
