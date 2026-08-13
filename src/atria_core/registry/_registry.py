from __future__ import annotations

import dataclasses
import difflib
import importlib
import pkgutil
from collections.abc import Callable, ItemsView
from typing import Any, Generic, TypeVar

from atria_core.registry._module_config import ModuleConfig

T = TypeVar("T", bound=ModuleConfig)


def import_submodules(package: str) -> None:
    module = importlib.import_module(package)
    if not hasattr(module, "__path__"):
        return
    for _, name, _ in pkgutil.walk_packages(
        module.__path__, prefix=f"{module.__name__}."
    ):
        importlib.import_module(name)


class ConfigRegistry(Generic[T]):
    """Maps names to config classes, so a config can be created by name.

    Entries registered with `register()` hold the class itself. Entries added
    from a serialized mapping with `from_dict()` hold an import path, and are
    imported the first time that name is looked up.

    What a created config is then used for is the caller's business; this only
    resolves a name into a validated config.

    Args:
        name: Name of this registry group, used in error messages.
        target_type: Config class every entry must subclass.
    """

    def __init__(self, name: str, target_type: type[T]) -> None:
        self.name = name
        self._target_type = target_type
        self._store: dict[str, type[T] | str] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator that registers the decorated class under `name`."""

        def decorator(target: type[T]) -> type[T]:
            if name in self._store:
                raise ValueError(
                    f"registry {self.name!r} already has an entry named {name!r} "
                    f"({self._describe(self._store[name])}) -- pick a different "
                    f"name, or remove the other registration"
                )
            self._store[name] = self._validated(target, name=name)
            return target

        return decorator

    def _get_class(self, name: str) -> type[T]:
        """Return the class registered under `name`, importing it if needed.

        Raises:
            KeyError: If nothing is registered under `name`.
            TypeError: If the entry does not subclass this registry's target type.
        """
        try:
            entry = self._store[name]
        except KeyError:
            raise KeyError(
                f"{self.name} has no entry named {name!r}."
                f"{self._suggest(name)} "
                f"Available: {', '.join(sorted(self.list())) or '<empty>'}"
            ) from None

        if not isinstance(entry, str):
            return entry

        resolved = self._validated(self._import(entry, name=name), name=name)
        self._store[name] = resolved
        return resolved

    def create(self, name: str, **params: Any) -> T:
        """Create an instance of the class registered under `name` from `params`.

        Intended for names and values arriving as data -- a command line flag,
        a config file -- where parameters cannot be checked statically. Code
        that knows which class it wants should import and construct it
        directly instead.

        Raises:
            KeyError: If nothing is registered under `name`.
            TypeError: If `params` holds names the class does not accept, or
                values it rejects.
        """
        target = self._get_class(name)
        self._check_params(target, params=params, name=name)
        try:
            return target(**params)
        except TypeError as e:
            raise TypeError(
                f"cannot create {name!r} ({target.__name__}) from "
                f"{self._describe_params(params)}: {e}"
            ) from e

    def items(self) -> ItemsView[str, type[T]]:
        """Return every name and its class, importing any entry not yet loaded."""
        return {name: self._get_class(name) for name in self._store}.items()

    def list(self) -> list[str]:
        """Return every registered name, without importing anything."""
        return list(self._store.keys())

    def to_dict(self) -> dict[str, str]:
        """Return the registry as a mapping of name to import path."""
        return {
            name: entry
            if isinstance(entry, str)
            else f"{entry.__module__}.{entry.__qualname__}"
            for name, entry in self._store.items()
        }

    def from_dict(self, data: dict[str, str]) -> None:
        """Add entries from a mapping of name to import path.

        Nothing is imported here; each path is resolved and checked the first
        time its name is looked up.
        """
        self._store.update(data)

    def _import(self, path: str, *, name: str) -> Any:
        module_name, _, attr = path.rpartition(".")
        if not module_name:
            raise ValueError(
                f"registry {self.name!r} entry {name!r} is {path!r}, which is not "
                f"a 'module.ClassName' import path"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"registry {self.name!r} entry {name!r} points at {path!r}, but "
                f"module {module_name!r} could not be imported: {e}"
            ) from e
        try:
            return getattr(module, attr)
        except AttributeError:
            raise AttributeError(
                f"registry {self.name!r} entry {name!r} points at {path!r}, but "
                f"module {module_name!r} has no attribute {attr!r}"
            ) from None

    def _validated(self, target: Any, *, name: str) -> type[T]:
        """Check `target` against this registry's config type and return it typed."""
        target_type = self._target_type
        if not isinstance(target, type):
            raise TypeError(
                f"registry {self.name!r} only accepts classes, but {name!r} is "
                f"{target!r} ({type(target).__name__})"
            )
        if not issubclass(target, target_type):
            bases = ", ".join(base.__name__ for base in target.__bases__)
            raise TypeError(
                f"registry {self.name!r} only accepts subclasses of "
                f"{target_type.__name__}, but {name!r} is {target.__name__}, "
                f"which subclasses {bases}"
            )
        return target

    def _check_params(
        self, target: type[T], *, params: dict[str, Any], name: str
    ) -> None:
        if not dataclasses.is_dataclass(target):
            return
        accepted = {field.name for field in dataclasses.fields(target)}
        unknown = sorted(set(params) - accepted)
        if unknown:
            raise TypeError(
                f"{name!r} ({target.__name__}) does not accept "
                f"{', '.join(repr(field) for field in unknown)}. "
                f"Accepted parameters: {', '.join(sorted(accepted)) or '<none>'}"
            )

    def _suggest(self, name: str) -> str:
        """Return a ' Did you mean ...?' hint for the closest registered names."""
        close_matches = difflib.get_close_matches(name, self.list(), n=3)
        if not close_matches:
            return ""
        return f" Did you mean {' or '.join(repr(m) for m in close_matches)}?"

    def _describe(self, entry: type[T] | str) -> str:
        return entry if isinstance(entry, str) else entry.__qualname__

    def _describe_params(self, params: dict[str, Any]) -> str:
        if not params:
            return "no parameters"
        return ", ".join(f"{key}={value!r}" for key, value in sorted(params.items()))
