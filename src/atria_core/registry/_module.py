from __future__ import annotations

from typing import Any, ClassVar, Self, TypeVar, cast, get_args, get_origin

from atria_core.registry._module_config import ModuleConfig


def _concrete_config_class(value: Any) -> type[ModuleConfig] | None:
    """Resolve `value` to a concrete `ModuleConfig` subclass, if possible.

    `value` is either the config type itself, or a type var standing in for
    it -- resolved through the type var's default (falling back to its
    bound), the way an unsubscripted generic base leaves it.
    """
    if isinstance(value, TypeVar):
        default = getattr(value, "__default__", None)
        if isinstance(default, type) and issubclass(default, ModuleConfig):
            return default
        value = value.__bound__

    if (
        isinstance(value, type)
        and issubclass(value, ModuleConfig)
        and value is not ModuleConfig
    ):
        return value
    return None


def _find_config_class(module_cls: type[Any]) -> type[ModuleConfig] | None:
    """Resolve the config type bound via `Module[MyConfig]` anywhere in
    `module_cls`'s generic base hierarchy.

    Walks every generic base recursively, substituting each base's own type
    vars with the args it was given -- so a config bound several generic
    levels up (e.g. through an intermediate `Dataset[X, MyConfig]` base) is
    still found, not just one bound directly on `Module`.
    """

    def resolve(cls: type[Any], type_vars: dict[Any, Any]) -> type[ModuleConfig] | None:
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base) or base
            args = tuple(type_vars.get(arg, arg) for arg in get_args(base))

            if origin is Module and args:
                config_cls = _concrete_config_class(args[0])
                if config_cls is not None:
                    return config_cls

            parameters = getattr(origin, "__parameters__", ())
            config_cls = resolve(origin, dict(zip(parameters, args, strict=False)))
            if config_cls is not None:
                return config_cls
        return None

    return resolve(module_cls, {})


class Module[T_ModuleConfig: ModuleConfig]:
    """Base class for objects built from a `ModuleConfig`.

    A concrete subclass binds its config type via `Module[MyConfig]` and
    declares a unique `__module_name__`. `__abstract__` opts a class out of
    both requirements, for bases meant to be subclassed rather than
    instantiated directly -- it is read from each class's own namespace, so
    it is never silently inherited by a concrete subclass.
    """

    __module_name__: ClassVar[str]
    __config_type__: ClassVar[type[ModuleConfig]]
    __abstract__: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__abstract__" not in cls.__dict__:
            cls.__abstract__ = False

        found = _find_config_class(cls)
        if found is not None:
            cls.__config_type__ = found
        elif not cls.__abstract__ and not hasattr(cls, "__config_type__"):
            raise TypeError(
                f"{cls.__name__} must bind a concrete ModuleConfig, "
                f"e.g. class {cls.__name__}(Module[MyConfig]): ..."
            )

        if not cls.__abstract__ and "__module_name__" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} is not abstract and must define a "
                f"'__module_name__' class variable."
            )

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls.__abstract__:
            raise TypeError(
                f"{cls.__name__} is abstract and cannot be instantiated directly. "
                f"Subclass it with a concrete ModuleConfig, e.g. "
                f"class My{cls.__name__}({cls.__name__}[MyParams]): ..."
            )
        return super().__new__(cls)

    def __init__(self, config: T_ModuleConfig | None = None) -> None:
        expected = type(self).__config_type__
        if config is None:
            config = cast("T_ModuleConfig", expected())
        elif not isinstance(config, expected):
            raise TypeError(
                f"{type(self).__name__} takes {expected.__name__}, "
                f"got {type(config).__name__}"
            )
        self._config: T_ModuleConfig = config

    @property
    def config(self) -> T_ModuleConfig:
        return self._config

    @property
    def name(self) -> str:
        """The name this module's class is registered under."""
        return type(self).__module_name__

    @classmethod
    def config_type(cls) -> type[T_ModuleConfig]:
        """Return the config type bound via `Module[MyConfig]`."""
        return cast("type[T_ModuleConfig]", cls.__config_type__)
