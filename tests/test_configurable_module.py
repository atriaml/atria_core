import pytest

from atria_core.registry import ConfigurableModule


def test_missing_config_type_raises_instead_of_falling_back() -> None:
    class MissingConfigModule(ConfigurableModule):  # type: ignore[type-arg]
        pass

    with pytest.raises(
        TypeError,
        match=(
            "Invalid config type for MissingConfigModule: expected a concrete "
            "subclass of ModuleConfig, but got no config type"
        ),
    ):
        MissingConfigModule.config_class()


def test_invalid_config_type_reports_actual_type() -> None:
    class InvalidConfigModule(ConfigurableModule[int]):  # type: ignore[type-var]
        pass

    with pytest.raises(
        TypeError,
        match=(
            "Invalid config type for InvalidConfigModule: expected a concrete "
            "subclass of ModuleConfig, but got int"
        ),
    ):
        InvalidConfigModule.config_class()
