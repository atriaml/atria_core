from __future__ import annotations

from atria_core.registry import import_submodules


def test_import_submodules_prefills_registry_without_manual_imports() -> None:
    from tests.registry._discovery_fixtures import items

    assert items.list() == []

    import_submodules("tests.registry._discovery_fixtures")

    assert sorted(items.list()) == ["a", "b"]
