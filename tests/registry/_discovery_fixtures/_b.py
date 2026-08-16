from __future__ import annotations

from tests.registry._discovery_fixtures import Item, items


@items.register
class B(Item):
    __module_name__ = "b"
