# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._configurable_module import ConfigurableModule as ConfigurableModule
    from ._module_config import ModuleConfig as ModuleConfig

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_configurable_module": ["ConfigurableModule"],
        "_module_config": ["ModuleConfig"],
    },
)
