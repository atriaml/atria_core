from __future__ import annotations

import enum


class BoundingBoxMode(str, enum.Enum):
    XYXY = "xyxy"
    XYWH = "xywh"


class BoundingBox:
    def __init__(
        self,
        value: list[float],
        mode: BoundingBoxMode = BoundingBoxMode.XYXY,
        normalized: bool = False,
    ) -> None:
        self.value = list(value)
        self.mode = BoundingBoxMode(mode) if isinstance(mode, str) else mode
        self.normalized = normalized

    # -------------------------------------
    # Basic attributes
    # -------------------------------------
    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def x1(self) -> float:
        return self.value[0]

    @property
    def y1(self) -> float:
        return self.value[1]

    @property
    def x2(self) -> float:
        return (
            self.x1 + self.width if self.mode == BoundingBoxMode.XYWH else self.value[2]
        )

    @property
    def y2(self) -> float:
        return (
            self.y1 + self.height
            if self.mode == BoundingBoxMode.XYWH
            else self.value[3]
        )

    @property
    def width(self) -> float:
        return self.value[2] if self.mode == BoundingBoxMode.XYWH else self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.value[3] if self.mode == BoundingBoxMode.XYWH else self.y2 - self.y1

    # -------------------------------------
    # Ops (formerly BoundingBoxOps)
    # -------------------------------------
    def switch_mode(self) -> BoundingBox:
        """Switches the bounding box mode between XYXY and XYWH."""
        if self.mode == BoundingBoxMode.XYXY:
            return BoundingBox(
                [self.x1, self.y1, self.width, self.height],
                BoundingBoxMode.XYWH,
                self.normalized,
            )
        return BoundingBox(
            [self.x1, self.y1, self.x2, self.y2], BoundingBoxMode.XYXY, self.normalized
        )

    def normalize(self, width: float, height: float) -> BoundingBox:
        """Normalizes coordinates to [0, 1]."""

        def clip(v: float) -> float:
            return min(max(v, 0.0), 1.0)

        if self.mode == BoundingBoxMode.XYWH:
            value = [
                clip(self.x1 / width),
                clip(self.y1 / height),
                clip(self.width / width),
                clip(self.height / height),
            ]
        else:
            value = [
                clip(self.x1 / width),
                clip(self.y1 / height),
                clip(self.x2 / width),
                clip(self.y2 / height),
            ]
        return BoundingBox(value, self.mode, normalized=True)

    def unnormalize(self, width: float, height: float) -> BoundingBox:
        """Unnormalizes coordinates to absolute pixel values."""
        if not self.normalized:
            return self

        if self.mode == BoundingBoxMode.XYWH:
            value = [
                self.x1 * width,
                self.y1 * height,
                self.width * width,
                self.height * height,
            ]
        else:
            value = [
                self.x1 * width,
                self.y1 * height,
                self.x2 * width,
                self.y2 * height,
            ]
        return BoundingBox(value, self.mode, normalized=False)

    # -------------------------------------
    # Dunder helpers
    # -------------------------------------
    def copy(self, **updates) -> BoundingBox:
        return BoundingBox(
            value=updates.get("value", list(self.value)),
            mode=updates.get("mode", self.mode),
            normalized=updates.get("normalized", self.normalized),
        )

    def __repr__(self) -> str:
        return f"BoundingBox(value={self.value}, mode={self.mode.value}, normalized={self.normalized})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return (
            self.value == other.value
            and self.mode == other.mode
            and self.normalized == other.normalized
        )
