from __future__ import annotations

import numpy as np
from PIL.Image import Resampling

from atria_core.types._generic._image import Image


class ImageTransformer:
    """Pure functions for transforming an Image, grouped under one class.
    Each takes an Image and returns a new one, so calls compose directly:
    `ImageTransformer.to_grayscale(ImageTransformer.resize(image, 10, 5))`.
    """

    @staticmethod
    def resize(
        image: Image, width: int, height: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        return Image.from_source(image.require_content().resize((width, height), resample))

    @staticmethod
    def resize_with_aspect_ratio(
        image: Image, max_size: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        assert max_size > 0, "max_size must be > 0"

        if max(image.width, image.height) <= max_size:
            return image

        if image.width >= image.height:
            new_w = max_size
            new_h = int(image.height * (max_size / image.width))
        else:
            new_h = max_size
            new_w = int(image.width * (max_size / image.height))

        return ImageTransformer.resize(image, new_w, new_h, resample)

    @staticmethod
    def to_rgb(image: Image) -> Image:
        return Image.from_source(image.require_content().convert("RGB"))

    @staticmethod
    def to_grayscale(image: Image) -> Image:
        return Image.from_source(image.require_content().convert("L"))

    @staticmethod
    def to_numpy(image: Image) -> np.ndarray:
        return np.array(image.require_content())
