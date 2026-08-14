from __future__ import annotations

from typing import Any, Generic, TypeVar, cast, get_args, get_origin

from atria_core.registry._module_config import ModuleConfig

T_ModuleConfig = TypeVar("T_ModuleConfig", bound=ModuleConfig)


def _concrete_config_class(value: Any) -> type[ModuleConfig] | None:
    if isinstance(value, TypeVar):
        default = getattr(value, "__default__", None)
        if isinstance(default, type) and issubclass(default, ModuleConfig):
            return default
        value = value.__bound__
    if isinstance(value, type) and issubclass(value, ModuleConfig):
        return value
    return None


def _resolve_config_class(module_cls: type[Any]) -> type[ModuleConfig]:
    """Resolve the config bound to ``ConfigurableModule`` in a class hierarchy."""

    encountered_config_type: Any = None

    def resolve(
        cls: type[Any], type_vars: dict[TypeVar, Any]
    ) -> type[ModuleConfig] | None:
        nonlocal encountered_config_type
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base) or base
            args = tuple(type_vars.get(arg, arg) for arg in get_args(base))

            if origin is ConfigurableModule and args:
                encountered_config_type = args[0]
                config_cls = _concrete_config_class(args[0])
                if config_cls is not None:
                    encountered_config_type = config_cls
                    return config_cls

            parameters = getattr(origin, "__parameters__", ())
            config_cls = resolve(origin, dict(zip(parameters, args, strict=False)))
            if config_cls is not None:
                return config_cls
        return None

    config_cls = resolve(module_cls, {})
    if config_cls is None or config_cls is ModuleConfig:
        if encountered_config_type is None:
            actual_type = "no config type"
        elif isinstance(encountered_config_type, type):
            actual_type = encountered_config_type.__name__
        else:
            actual_type = repr(encountered_config_type)
        raise TypeError(
            f"Invalid config type for {module_cls.__name__}: expected a concrete "
            f"subclass of ModuleConfig, but got {actual_type}."
        )
    return config_cls


class ConfigurableModule(Generic[T_ModuleConfig]):
    """Base class for objects built from a ModuleConfig, exposing the config
    that built them as `self.config`."""

    @classmethod
    def config_class(cls) -> type[T_ModuleConfig]:
        """Return the config type declared by ``ConfigurableModule[Config]``."""
        return cast("type[T_ModuleConfig]", _resolve_config_class(cls))

    def __init__(self, config: T_ModuleConfig | None = None) -> None:
        expected_config_cls = type(self).config_class()
        if config is None:
            config = expected_config_cls()
        elif not isinstance(config, expected_config_cls):
            raise TypeError(
                f"{type(self).__name__} takes a {expected_config_cls.__name__}, "
                f"but got {type(config).__name__}"
            )
        self._config = config

    @property
    def config(self) -> T_ModuleConfig:
        return self._config
