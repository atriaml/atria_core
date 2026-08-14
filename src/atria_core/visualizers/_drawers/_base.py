from __future__ import annotations

from typing import Protocol, TypeVar

from atria_core.types._arrays import FloatArray
from atria_core.visualizers._drawers._style import DEFAULT_STYLE, DrawStyle

T = TypeVar("T")


class BboxDrawer(Protocol[T]):
    """Draws bboxes (+ optional per-box text/label) onto a target, using a
    shared DrawStyle for appearance. `bboxes` are always already in the
    target's native coordinate space (pixels for images, points for PDF
    pages) -- callers use ElementArray's coordinate metadata to convert
    boxes before calling draw()."""

    def draw(
        self,
        target: T,
        bboxes: FloatArray,
        *,
        texts: list[str] | None = None,
        labels: list[str] | None = None,
        style: DrawStyle = DEFAULT_STYLE,
    ) -> T: ...
