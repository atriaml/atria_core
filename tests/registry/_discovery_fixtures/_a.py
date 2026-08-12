from __future__ import annotations

from tests.registry._discovery_fixtures import items


@items.register("a")
class A:
    pass
