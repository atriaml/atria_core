from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np

from atria_core.visualizers._drawers._style import DEFAULT_STYLE, DrawStyle

T = TypeVar("T")


class BboxDrawer(Protocol[T]):
    """Draws bboxes (+ optional per-box text/label) onto a target, using a
    shared DrawStyle for appearance. `bboxes` are always already in the
    target's native coordinate space (pixels for images, points for PDF
    pages) -- callers scale from ElementArray's normalized [0,1] boxes
    before calling draw()."""

    def draw(
        self,
        target: T,
        bboxes: np.ndarray,
        *,
        texts: list[str] | None = None,
        labels: list[str] | None = None,
        style: DrawStyle = DEFAULT_STYLE,
    ) -> T: ...
