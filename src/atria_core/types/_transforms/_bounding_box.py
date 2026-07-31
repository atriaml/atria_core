from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np

from atria_core.types._generic._bounding_box import BoundingBoxMode

Container = TypeVar("Container", bound="BoxBatchOwner")


class BoxBatchOwner(Protocol):
    """Anything holding one or more named batches of bboxes, plus the metadata
    describing them as a whole (DocumentContent, ObjectDetectionAnnotation).
    This is what lets BoundingBoxTransformer work generically over either."""

    bbox_mode: BoundingBoxMode
    normalized: bool

    def box_batches(self) -> dict[str, tuple[np.ndarray, list[int]] | None]: ...

    def with_box_batches(
        self: Container,
        batches: dict[str, tuple[np.ndarray, list[int]]],
        *,
        normalized: bool,
        mode: BoundingBoxMode,
    ) -> Container: ...


class BoundingBoxTransformer:
    """Pure functions for transforming a bbox-batch-owning container
    (DocumentContent, ObjectDetectionAnnotation), grouped under one class.
    Each takes a container and returns a new one, so calls compose directly:
    `x = BoundingBoxTransformer.normalize(x, w, h)`
    `x = BoundingBoxTransformer.unnormalize(x, w, h)`
    `x = BoundingBoxTransformer.switch_mode(x)`
    """

    @staticmethod
    def normalize(container: Container, width: float, height: float) -> Container:
        if container.normalized:
            return container
        return BoundingBoxTransformer._apply(
            container,
            lambda boxes: np.clip(boxes / [width, height, width, height], 0.0, 1.0),
            normalized=True,
            mode=container.bbox_mode,
        )

    @staticmethod
    def unnormalize(container: Container, width: float, height: float) -> Container:
        if not container.normalized:
            return container
        return BoundingBoxTransformer._apply(
            container,
            lambda boxes: boxes * [width, height, width, height],
            normalized=False,
            mode=container.bbox_mode,
        )

    @staticmethod
    def switch_mode(container: Container) -> Container:
        mode = container.bbox_mode
        new_mode = (
            BoundingBoxMode.XYWH if mode == BoundingBoxMode.XYXY else BoundingBoxMode.XYXY
        )

        def _switch(boxes: np.ndarray) -> np.ndarray:
            x1, y1, a, b = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            if mode == BoundingBoxMode.XYXY:
                return np.stack([x1, y1, a - x1, b - y1], axis=1)
            return np.stack([x1, y1, x1 + a, y1 + b], axis=1)

        return BoundingBoxTransformer._apply(
            container, _switch, normalized=container.normalized, mode=new_mode
        )

    @staticmethod
    def _apply(
        container: Container,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        normalized: bool,
        mode: BoundingBoxMode,
    ) -> Container:
        batches = container.box_batches()
        transformed = {
            name: (fn(boxes), indices)
            for name, batch in batches.items()
            if batch is not None
            for boxes, indices in [batch]
        }
        if not transformed:
            return container
        return container.with_box_batches(transformed, normalized=normalized, mode=mode)
