from __future__ import annotations

import pytest

from atria_core.registry import Module, ModuleConfig


class DummyConfig(ModuleConfig):
    pass


def test_missing_config_type_raises_at_class_definition() -> None:
    with pytest.raises(
        TypeError, match="MissingConfigModule must bind a concrete ModuleConfig"
    ):

        class MissingConfigModule(Module):  # type: ignore[type-arg]
            pass


def test_invalid_config_type_raises_at_class_definition() -> None:
    with pytest.raises(
        TypeError, match="InvalidConfigModule must bind a concrete ModuleConfig"
    ):

        class InvalidConfigModule(Module[int]):  # type: ignore[type-var]
            pass


def test_concrete_module_without_a_name_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "UnnamedModule is not abstract and must define a "
            "'__module_name__' class variable"
        ),
    ):

        class UnnamedModule(Module[DummyConfig]):
            pass


def test_abstract_module_needs_neither_config_nor_name() -> None:
    class AbstractModule(Module):  # type: ignore[type-arg]
        __abstract__ = True

    assert AbstractModule.__abstract__ is True


def test_abstract_is_not_inherited_by_concrete_subclasses() -> None:
    class AbstractBase(Module[DummyConfig]):
        __abstract__ = True

    class ConcreteLeaf(AbstractBase):
        __module_name__ = "concrete-leaf"

    assert ConcreteLeaf.__abstract__ is False
    assert isinstance(ConcreteLeaf(), ConcreteLeaf)


def test_abstract_module_cannot_be_instantiated() -> None:
    class AbstractBase(Module[DummyConfig]):
        __abstract__ = True

    with pytest.raises(TypeError, match="AbstractBase is abstract"):
        AbstractBase()


def test_mismatched_config_is_rejected() -> None:
    class OtherConfig(ModuleConfig):
        pass

    class NamedModule(Module[DummyConfig]):
        __module_name__ = "named"

    with pytest.raises(TypeError, match="takes DummyConfig, got OtherConfig"):
        NamedModule(OtherConfig())  # type: ignore[arg-type]


def test_default_config_is_constructed_when_none_given() -> None:
    class NamedModule(Module[DummyConfig]):
        __module_name__ = "named-default"

    assert isinstance(NamedModule().config, DummyConfig)


def test_config_type_classmethod_returns_bound_config() -> None:
    class NamedModule(Module[DummyConfig]):
        __module_name__ = "named-config-type"

    assert NamedModule.config_type() is DummyConfig


def test_name_property_returns_module_name() -> None:
    class NamedModule(Module[DummyConfig]):
        __module_name__ = "named-property"

    assert NamedModule().name == "named-property"
