# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    # Self-aliased (`as Name`) so mypy's --no-implicit-reexport (part of
    # `strict`) treats these as explicit re-exports -- required for any
    # `from atria_core.registry import X` done outside this package, since
    # `__all__` below is computed at runtime by lazy_loader and isn't
    # visible to mypy as a literal list.
    from ._configurable_module import ConfigurableModule as ConfigurableModule
    from ._module_config import ModuleConfig as ModuleConfig
    from ._registry import Registry as Registry
    from ._registry_group import RegistryGroup as RegistryGroup
    from ._registry_store import RegistryStore as RegistryStore

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_configurable_module": ["ConfigurableModule"],
        "_module_config": ["ModuleConfig"],
        "_registry": ["Registry"],
        "_registry_group": ["RegistryGroup"],
        "_registry_store": ["RegistryStore"],
    },
)
