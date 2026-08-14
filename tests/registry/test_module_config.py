from __future__ import annotations

import enum
from pathlib import Path

import pytest
from pydantic import ValidationError

from atria_core.registry import ModuleConfig


class Precision(enum.Enum):
    FP16 = "fp16"
    FP32 = "fp32"


class OptimizerConfig(ModuleConfig):
    lr: float = 1e-3
    precision: Precision = Precision.FP32


class SgdConfig(OptimizerConfig):
    momentum: float = 0.9


class TrainerConfig(ModuleConfig):
    optimizer: SgdConfig = SgdConfig()
    epochs: int = 10


def test_subclass_needs_no_decorator_to_validate_fields() -> None:
    with pytest.raises(ValidationError, match="lr"):
        OptimizerConfig(lr="fast")  # type: ignore[arg-type]


def test_config_is_frozen() -> None:
    config = OptimizerConfig()

    with pytest.raises(ValidationError, match="frozen"):
        config.lr = 0.5


def test_unknown_param_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OptimizerConfig(nonsense=1)  # type: ignore[call-arg]


def test_to_dict_is_json_safe() -> None:
    data = SgdConfig(lr=0.01, precision=Precision.FP16).to_dict()

    assert data == {"lr": 0.01, "precision": "fp16", "momentum": 0.9}


def test_config_round_trips_through_to_dict_from_dict() -> None:
    config = SgdConfig(lr=0.01, precision=Precision.FP16)

    assert SgdConfig.from_dict(config.to_dict()) == config


def test_nested_config_round_trips_through_to_dict_from_dict() -> None:
    config = TrainerConfig(optimizer=SgdConfig(lr=0.01, momentum=0.99), epochs=3)

    restored = TrainerConfig.from_dict(config.to_dict())

    assert isinstance(restored.optimizer, SgdConfig)
    assert restored == config


def test_from_dict_rejects_a_wrong_typed_field() -> None:
    data = OptimizerConfig().to_dict()
    data["lr"] = "fast"

    with pytest.raises(ValidationError) as excinfo:
        OptimizerConfig.from_dict(data)

    assert excinfo.value.errors()[0]["loc"] == ("lr",)


def test_from_dict_rejects_an_unknown_field() -> None:
    data = OptimizerConfig().to_dict()
    data["typo_field"] = 1

    with pytest.raises(ValidationError) as excinfo:
        OptimizerConfig.from_dict(data)

    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"


def test_hash_is_stable_for_equal_params() -> None:
    assert SgdConfig(lr=0.01).hash == SgdConfig(lr=0.01).hash


def test_hash_differs_for_different_params() -> None:
    assert SgdConfig(lr=0.01).hash != SgdConfig(lr=0.02).hash


def test_non_json_safe_field_is_rejected_at_class_definition() -> None:
    with pytest.raises(TypeError, match="which is not JSON-safe"):

        class BadConfig(ModuleConfig):
            output_dir: Path = Path("/tmp")


def test_list_of_configs_field_is_rejected_at_class_definition() -> None:
    with pytest.raises(TypeError, match="which is not JSON-safe"):

        class BadConfig(ModuleConfig):
            optimizers: list[OptimizerConfig] = []


def test_containers_of_primitives_are_accepted() -> None:
    class GoodConfig(ModuleConfig):
        names: list[str] = []
        limits: dict[str, int] = {}
        betas: tuple[float, float] = (0.9, 0.999)

    assert GoodConfig().to_dict() == {
        "names": [],
        "limits": {},
        "betas": [0.9, 0.999],
    }
