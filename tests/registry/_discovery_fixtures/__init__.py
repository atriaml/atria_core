from __future__ import annotations

from atria_core.registry import ConfigRegistry

items: ConfigRegistry[object] = ConfigRegistry(
    "tests.registry._discovery_fixtures.items", target_type=object
)
