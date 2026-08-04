from __future__ import annotations

from dataclasses import is_dataclass

from atria_core.datasets._cacher import transform_hash
from atria_core.transforms import BaseTransform


class _PrefixTransform(BaseTransform):
    prefix: str
    sizes: tuple[int, int] = (10, 20)

    def __call__(self, value: str) -> str:
        return f"{self.prefix}{value}"


def test_base_transform_dumps_dataclass_config() -> None:
    transform = _PrefixTransform(prefix="test-")

    assert is_dataclass(transform)
    assert transform.dump() == {
        "type": f"{_PrefixTransform.__module__}.{_PrefixTransform.__qualname__}",
        "params": {"prefix": "test-", "sizes": [10, 20]},
    }
    assert transform.to_dict() == transform.dump()


def test_base_transform_hash_is_derived_from_config() -> None:
    first = _PrefixTransform(prefix="first-")
    same = _PrefixTransform(prefix="first-")
    different = _PrefixTransform(prefix="different-")

    assert first.hash == same.hash
    assert first.hash != different.hash
    assert transform_hash(first) == first.hash
