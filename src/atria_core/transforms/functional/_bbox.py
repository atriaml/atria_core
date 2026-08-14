from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np

from atria_core.types import BoundingBoxMode
from atria_core.types._arrays import FloatArray

Container = TypeVar("Container", bound="BoxBatchOwner")


class BoxBatchOwner(Protocol):
    """Anything holding one or more named batches of bboxes, plus the metadata
    describing them as a whole (DocumentContent, ObjectDetectionAnnotation).
    This is what lets these functions work generically over either.

    bbox_mode/normalized are declared as read-only properties, not plain
    attributes -- implementers are frozen dataclasses, so a plain attribute
    declaration (which Protocol treats as read+write) wouldn't structurally
    match; nothing here ever assigns to them anyway, only with_box_batches()
    produces a new instance."""

    @property
    def bbox_mode(self) -> BoundingBoxMode: ...

    @property
    def normalized(self) -> bool: ...

    def box_batches(self) -> dict[str, FloatArray | None]: ...

    def with_box_batches(
        self: Container,
        batches: dict[str, FloatArray],
        *,
        normalized: bool,
        mode: BoundingBoxMode,
    ) -> Container: ...


def normalize(container: Container, width: float, height: float) -> Container:
    if container.normalized:
        return container
    return _apply(
        container,
        lambda boxes: np.clip(boxes / [width, height, width, height], 0.0, 1.0),
        normalized=True,
        mode=container.bbox_mode,
    )


def unnormalize(container: Container, width: float, height: float) -> Container:
    if not container.normalized:
        return container
    return _apply(
        container,
        lambda boxes: boxes * [width, height, width, height],
        normalized=False,
        mode=container.bbox_mode,
    )


def switch_mode(container: Container) -> Container:
    mode = container.bbox_mode
    new_mode = (
        BoundingBoxMode.XYWH if mode == BoundingBoxMode.XYXY else BoundingBoxMode.XYXY
    )

    def _switch(boxes: FloatArray) -> FloatArray:
        x1, y1, a, b = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        if mode == BoundingBoxMode.XYXY:
            return np.stack([x1, y1, a - x1, b - y1], axis=1)
        return np.stack([x1, y1, x1 + a, y1 + b], axis=1)

    return _apply(container, _switch, normalized=container.normalized, mode=new_mode)


def _apply(
    container: Container,
    fn: Callable[[FloatArray], FloatArray],
    *,
    normalized: bool,
    mode: BoundingBoxMode,
) -> Container:
    batches = container.box_batches()
    transformed = {
        name: fn(boxes) for name, boxes in batches.items() if boxes is not None
    }
    if not transformed:
        return container
    return container.with_box_batches(transformed, normalized=normalized, mode=mode)
