from __future__ import annotations

import enum

import numpy as np


class BoundingBoxMode(str, enum.Enum):
    XYXY = "xyxy"
    XYWH = "xywh"


def as_bbox_array(value: object) -> np.ndarray:
    """Coerces raw input (e.g. JSON lists from from_dict) into a bbox array:
    shape (4,) float. Boundary-only conversion -- fields typed np.ndarray
    should be validated with check_bbox_array, not re-coerced."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"bbox must have shape (4,), got {arr.shape}")
    return arr


def check_bbox_array(value: np.ndarray) -> None:
    """Validates an already-constructed bbox array: shape (4,). Raises
    instead of coercing -- the field is typed np.ndarray, so a caller
    passing something else is a type error, not a value to convert."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"bbox must be a numpy array, got {type(value).__name__}")
    if value.shape != (4,):
        raise ValueError(f"bbox must have shape (4,), got {value.shape}")


def as_segmentation_array(value: object) -> np.ndarray:
    """Coerces raw input into a segmentation array: shape (P, 2) float.
    Boundary-only -- see as_bbox_array."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"segmentation must have shape (P, 2), got {arr.shape}")
    return arr


def check_segmentation_array(value: np.ndarray) -> None:
    """Validates an already-constructed segmentation array: shape (P, 2).
    See check_bbox_array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"segmentation must be a numpy array, got {type(value).__name__}")
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"segmentation must have shape (P, 2), got {value.shape}")
