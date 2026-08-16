# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._module import Module as Module
    from ._module_config import ModuleConfig as ModuleConfig
    from ._registry import ModuleRegistry as ModuleRegistry
    from ._registry import import_submodules as import_submodules
    from ._registry_store import RegistryStore as RegistryStore

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_module": ["Module"],
        "_module_config": ["ModuleConfig"],
        "_registry": ["ModuleRegistry", "import_submodules"],
        "_registry_store": ["RegistryStore"],
    },
)
