from __future__ import annotations

from typing import Generic, TypeVar

from atria_core.registry._module_config import ModuleConfig

T_ModuleConfig = TypeVar("T_ModuleConfig", bound=ModuleConfig)


class ConfigurableModule(Generic[T_ModuleConfig]):
    """Base class for objects built from a ModuleConfig, exposing the config
    that built them as `self.config`."""

    def __init__(self, config: T_ModuleConfig) -> None:
        self._config = config

    @property
    def config(self) -> T_ModuleConfig:
        return self._config
