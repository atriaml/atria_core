from __future__ import annotations

from typing import Any, ClassVar, Self, cast, get_args

from atria_core.registry._module_config import ModuleConfig


def _find_config_class(cls: type[Any]) -> type[ModuleConfig] | None:
    for base in cls.__dict__.get("__orig_bases__", ()):
        for arg in get_args(base):
            if (
                isinstance(arg, type)
                and issubclass(arg, ModuleConfig)
                and arg is not ModuleConfig
            ):
                return arg
    return None


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
