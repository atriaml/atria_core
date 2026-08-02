from __future__ import annotations

from collections.abc import ItemsView

from atria_core.registry._registry_group import RegistryGroup


class Registry:
    """Owns every RegistryGroup -- the one place that knows the full set."""

    _groups: dict[str, RegistryGroup] = {}

    @classmethod
    def group(cls, name: str) -> RegistryGroup:
        return cls._groups.setdefault(name, RegistryGroup(name))

    @classmethod
    def groups(cls) -> ItemsView[str, RegistryGroup]:
        return cls._groups.items()

    @classmethod
    def to_dict(cls) -> dict[str, dict[str, str]]:
        """Every registered config's import path, keyed by group then name."""
        return {
            group_name: {
                name: f"{config_cls.__module__}.{config_cls.__qualname__}"
                for name, config_cls in group.items()
            }
            for group_name, group in cls.groups()
        }
