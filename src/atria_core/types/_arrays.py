from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
"""Array of 64-bit floats: coordinates, confidences, angles."""

type IntArray = NDArray[np.int_]
"""Array of platform-width integers: identifiers, levels, counts."""

type ObjectArray = NDArray[np.object_]
"""Array of Python objects, used for variable-length strings."""

type BoolArray = NDArray[np.bool_]
"""Array of booleans, used for per-element flags and masks."""

type ImageArray = NDArray[np.uint8]
"""Image pixels as 8-bit channels, the layout PIL images convert to."""
