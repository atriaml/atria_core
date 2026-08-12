from __future__ import annotations

from tests.registry._discovery_fixtures import items


@items.register("b")
class B:
    pass
