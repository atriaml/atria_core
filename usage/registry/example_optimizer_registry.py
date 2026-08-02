"""Example: registering configs, building modules, and round-tripping a
nested/polymorphic config through to_dict()/from_dict()."""

from __future__ import annotations

import enum

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.logger import get_logger
from atria_core.registry import ConfigurableModule, ModuleConfig, Registry

logger = get_logger(__name__)

optimizers = Registry.group("optimizers")


class Precision(enum.Enum):
    FP16 = "fp16"
    FP32 = "fp32"


@pydantic_dataclass(frozen=True)
class OptimizerConfig(ModuleConfig):
    """Never registered/instantiated directly -- only concrete subclasses
    (SgdConfig, AdamConfig) are."""

    lr: float = 1e-3
    precision: Precision = Precision.FP32


@optimizers.register("sgd")
@pydantic_dataclass(frozen=True)
class SgdConfig(OptimizerConfig):
    momentum: float = 0.9

    def build_module(self) -> Sgd:
        return Sgd(self)


class Sgd(ConfigurableModule[SgdConfig]):
    def __repr__(self) -> str:
        return f"Sgd(config={self.config})"


@optimizers.register("adam")
@pydantic_dataclass(frozen=True)
class AdamConfig(OptimizerConfig):
    betas: tuple[float, float] = (0.9, 0.999)

    def build_module(self) -> Adam:
        return Adam(self)


class Adam(ConfigurableModule[AdamConfig]):
    def __repr__(self) -> str:
        return f"Adam(config={self.config})"


@pydantic_dataclass(frozen=True)
class TrainerConfig(ModuleConfig):
    # holds either an SgdConfig or an AdamConfig -- no declared Union/discriminator.
    optimizer: OptimizerConfig
    epochs: int = 10

    def build_module(self) -> Trainer:
        return Trainer(self)


class Trainer(ConfigurableModule[TrainerConfig]):
    def __repr__(self) -> str:
        return f"Trainer(config={self.config})"


def main() -> None:
    sgd = SgdConfig(lr=0.01, momentum=0.99, precision=Precision.FP16).build_module()
    logger.info("Built: %s", sgd)

    trainer_cfg = TrainerConfig(optimizer=SgdConfig(lr=0.01, momentum=0.99))
    data = trainer_cfg.to_dict()
    logger.info("Serialized: %s", data)

    restored = TrainerConfig.from_dict(data)
    logger.info("Restored optimizer type: %s", type(restored.optimizer).__name__)

    # Swap the optimizer by editing the dict -- no schema change needed, hydra
    # resolves each nested dict via its own "_target_" key.
    data["optimizer"] = AdamConfig(lr=0.02).to_dict()
    restored = TrainerConfig.from_dict(data)
    logger.info(
        "Restored optimizer type after swap: %s", type(restored.optimizer).__name__
    )


if __name__ == "__main__":
    main()
