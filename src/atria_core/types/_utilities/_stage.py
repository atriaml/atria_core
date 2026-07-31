from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

In = TypeVar("In", contravariant=True)
Out = TypeVar("Out", covariant=True)


@runtime_checkable
class Stage(Protocol[In, Out]):
    """A typed, composable unit of transformation: given a value, returns a new value.

    All data transformations in this library (extractors, encoders, future
    preprocessing steps) are expected to conform to this shape as callable
    objects, rather than living as bare module-level functions. Cross-cutting
    concerns (I/O, logging, run parameters) belong in separate collaborator
    classes, not folded into the stage itself.
    """

    def __call__(self, value: In) -> Out: ...
