from __future__ import annotations

from atria_core.registry import Registry

items: Registry[type[object]] = Registry("tests.registry._discovery_fixtures.items")
