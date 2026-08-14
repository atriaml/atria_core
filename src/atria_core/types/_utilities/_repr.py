from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.repr import RichReprResult


class _ArraySummary:
    def __init__(self, value: object) -> None:
        array = cast(Any, value)
        self.shape = array.shape
        self.dtype = array.dtype

    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype})"


class RepresentationMixin:
    """
    Mixin class for rich representation of objects.

    This class provides methods for generating string representations of objects
    using the `rich` library. It includes support for both developer-friendly
    (`__repr__`) and human-readable (`__str__`) representations.

    Set `__repr_fields__` as a class attribute to specify which fields to include.
    Prefer a tuple when field order matters.
    """

    __repr_fields__: ClassVar[Collection[str]] = ()

    def __repr_name__(self) -> str:
        """
        Returns the name of the class for use in the `__repr__` method.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

    def __rich_repr__(self) -> RichReprResult:
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs for the specified fields only.
        """
        import types

        repr_fields: Iterable[str] = self.__repr_fields__ or self.__dict__.keys()

        for field_name in repr_fields:
            if not hasattr(self, field_name):
                continue

            # do not add private fields
            if field_name.startswith("_"):
                continue

            value = getattr(self, field_name)
            safe_value: Any
            if isinstance(value, types.MethodType):
                safe_value = value.__func__
            else:
                safe_value = value

            if value is None:
                yield field_name, safe_value, None
                continue

            # Array contents quickly dominate nested data-model reprs. Shape
            # and dtype carry the useful structural information without
            # dumping every coordinate, label, or pixel.
            if type(value).__module__.startswith("numpy") and hasattr(value, "shape"):
                safe_value = _ArraySummary(value)

            yield field_name, safe_value

    def __repr__(self) -> str:
        """
        Generates a developer-friendly string representation of the object.

        Returns:
            str: A developer-friendly string representation of the object.
        """

        from rich.pretty import pretty_repr

        from atria_core.types._constants import _MAX_REPR_PRINT_ELEMENTS

        formatted: str = pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=128, max_depth=8
        )
        return formatted

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the object.

        Returns:
            str: A human-readable string representation of the object.
        """

        from rich.pretty import pretty_repr

        from atria_core.types._constants import _MAX_REPR_PRINT_ELEMENTS

        formatted: str = pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=128, max_depth=8
        )
        return formatted
