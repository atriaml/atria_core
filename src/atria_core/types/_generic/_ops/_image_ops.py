from __future__ import annotations

import numpy as np
from PIL.Image import Resampling

from atria_core.logger import get_logger
from atria_core.types._base._ops._base_ops import StandardOps
from atria_core.types._generic._image import Image
from atria_core.types._pydantic import ValidatedPILImage

logger = get_logger(__name__)


class ImageOps(StandardOps[Image]):
    @property
    def content(self) -> ValidatedPILImage:
        return self.model.content

    def to_numpy(self) -> np.ndarray:
        return np.array(self.content)

    # -----------------------------
    # Color space conversions
    # -----------------------------
    def to_rgb(self) -> Image:
        return self.model.model_copy(update={"content": self.content.convert("RGB")})

    def to_grayscale(self) -> Image:
        return self.model.model_copy(update={"content": self.content.convert("L")})

    # -----------------------------
    # Resizing
    # -----------------------------
    def resize(
        self, width: int, height: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        return self.model.model_copy(
            update={"content": self.content.resize((width, height), resample)}
        )

    def resize_with_aspect_ratio(
        self, max_size: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        assert max_size > 0, "max_size must be > 0"

        width, height = self.model.size
        if max(width, height) <= max_size:
            return self.model

        if width >= height:
            new_w = max_size
            new_h = int(height * (max_size / width))
        else:
            new_h = max_size
            new_w = int(width * (max_size / height))

        return self.resize(new_w, new_h, resample)
