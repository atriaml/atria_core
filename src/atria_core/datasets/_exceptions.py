from __future__ import annotations


class ConfigurationNotFoundError(ValueError):
    def __init__(self, config_name: str, available_configs: list[str]) -> None:
        super().__init__(
            f"Configuration '{config_name}' not found in the dataset. "
            f"Available configurations: {', '.join(available_configs)}"
        )


class SplitNotFoundError(Exception):
    pass
