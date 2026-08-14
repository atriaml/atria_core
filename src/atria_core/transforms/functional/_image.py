from __future__ import annotations

import numpy as np
from PIL.Image import Resampling

from atria_core.types import Image
from atria_core.types._arrays import ImageArray


def resize(
    image: Image, width: int, height: int, resample: Resampling = Resampling.BICUBIC
) -> Image:
    return Image.from_source(image.require_content().resize((width, height), resample))


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

    return resize(image, new_w, new_h, resample)


def to_rgb(image: Image) -> Image:
    return Image.from_source(image.require_content().convert("RGB"))


def to_grayscale(image: Image) -> Image:
    return Image.from_source(image.require_content().convert("L"))


def to_numpy(image: Image) -> ImageArray:
    return np.array(image.require_content())
