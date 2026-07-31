from __future__ import annotations

import enum

import numpy as np


class BoundingBoxMode(str, enum.Enum):
    XYXY = "xyxy"
    XYWH = "xywh"


def as_bbox_array(value: object) -> np.ndarray:
    """A single bounding box: shape (4,) float array. Validated on construction."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"bbox must have shape (4,), got {arr.shape}")
    return arr


def as_segmentation_array(value: object) -> np.ndarray:
    """One object's segmentation polygon: shape (P, 2) float array of points."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"segmentation must have shape (P, 2), got {arr.shape}")
    return arr
