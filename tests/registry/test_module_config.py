from __future__ import annotations

import dataclasses
import enum
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.registry import ConfigurableModule, ModuleConfig, Registry

_segmentors: Registry[type[QuickshiftConfig]] = Registry("test_module_config.segmentors")
_optimizers: Registry[type[OptimizerConfig]] = Registry("test_module_config.optimizers")


@_segmentors.register("quickshift")
@pydantic_dataclass(frozen=True)
class QuickshiftConfig(ModuleConfig):
    kernel_size: int = 4
    max_dist: int = 200
    ratio: float = 0.2

    def build_module(self) -> QuickshiftSegmenter:
        return QuickshiftSegmenter(self)


class QuickshiftSegmenter(ConfigurableModule[QuickshiftConfig]):
    pass


class Precision(enum.Enum):
    FP16 = "fp16"
    FP32 = "fp32"


@pydantic_dataclass(frozen=True)
class OptimizerConfig(ModuleConfig):

    lr: float = 1e-3
    precision: Precision = Precision.FP32


class _FakeAdam:
    def __init__(
        self, lr: float, precision: Precision, betas: tuple[float, float]
    ) -> None:
        self.lr = lr
        self.precision = precision
        self.betas = betas


@_optimizers.register("sgd")
@pydantic_dataclass(frozen=True)
class SgdConfig(OptimizerConfig):
    momentum: float = 0.9

    def build_module(self) -> Sgd:
        return Sgd(self)


class Sgd(ConfigurableModule[SgdConfig]):
    pass


@_optimizers.register("adam")
@pydantic_dataclass(frozen=True)
class AdamConfig(OptimizerConfig):
    betas: tuple[float, float] = (0.9, 0.999)

    def build_module(self) -> _FakeAdam:
        return _FakeAdam(lr=self.lr, precision=self.precision, betas=self.betas)


@_optimizers.register("trainer")
@pydantic_dataclass(frozen=True)
class TrainerConfig(ModuleConfig):
    optimizer: OptimizerConfig
    epochs: int = 10

    def build_module(self) -> Trainer:
        return Trainer(self)


class Trainer(ConfigurableModule[TrainerConfig]):
    pass


def test_build_module_constructs_configurable_module_directly() -> None:
    module = QuickshiftConfig(kernel_size=8, max_dist=100, ratio=0.3).build_module()

    assert isinstance(module, QuickshiftSegmenter)
    assert module.config == QuickshiftConfig(kernel_size=8, max_dist=100, ratio=0.3)


def test_build_module_on_third_party_class_with_custom_unpacking() -> None:
    adam = AdamConfig(lr=0.02, precision=Precision.FP16).build_module()

    assert isinstance(adam, _FakeAdam)
    assert adam.lr == 0.02
    assert adam.precision is Precision.FP16
    assert adam.betas == (0.9, 0.999)


def test_base_module_config_build_module_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        OptimizerConfig().build_module()


def test_nested_polymorphic_config_round_trips_via_to_dict_from_dict() -> None:
    trainer_cfg = TrainerConfig(
        optimizer=SgdConfig(lr=0.01, momentum=0.99, precision=Precision.FP16)
    )

    data = trainer_cfg.to_dict()
    restored = TrainerConfig.from_dict(data)

    assert isinstance(restored, TrainerConfig)
    assert isinstance(restored.optimizer, SgdConfig)
    # pydantic coerces the raw "fp16" string back into the Precision enum.
    assert restored.optimizer.precision is Precision.FP16


def test_swap_polymorphic_field_by_editing_dict() -> None:
    trainer_cfg = TrainerConfig(optimizer=SgdConfig())
    data = trainer_cfg.to_dict()

    data["optimizer"] = AdamConfig(lr=0.02, precision=Precision.FP32).to_dict()
    restored = TrainerConfig.from_dict(data)

    assert isinstance(restored, TrainerConfig)
    assert isinstance(restored.optimizer, AdamConfig)


def test_list_of_configs_field_is_rejected() -> None:
    @pydantic_dataclass(frozen=True)
    class BadConfig(ModuleConfig):
        optimizers: list[OptimizerConfig] = dataclasses.field(default_factory=list)

        def build_module(self) -> Any:
            raise NotImplementedError

    with pytest.raises(ValidationError):
        BadConfig(optimizers=[SgdConfig()])


def test_plain_primitive_and_nested_config_fields_are_accepted() -> None:
    # Doesn't raise -- ints/floats/tuples/nested ModuleConfig/enum are all JSON-safe.
    TrainerConfig(optimizer=AdamConfig(betas=(0.1, 0.2)), epochs=5)
