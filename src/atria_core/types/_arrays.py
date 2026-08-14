from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
"""Array of 64-bit floats: coordinates, confidences, angles."""

IntArray: TypeAlias = NDArray[np.int_]
"""Array of platform-width integers: identifiers, levels, counts."""

ObjectArray: TypeAlias = NDArray[np.object_]
"""Array of Python objects, used for variable-length strings."""

BoolArray: TypeAlias = NDArray[np.bool_]
"""Array of booleans, used for per-element flags and masks."""

ImageArray: TypeAlias = NDArray[np.uint8]
"""Image pixels as 8-bit channels, the layout PIL images convert to."""
