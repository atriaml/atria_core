from __future__ import annotations

from PIL import Image as PILImage

from atria_core.types._generic._image import Image
from atria_core.types._transforms._image import ImageTransformer


def test_resize_exact() -> None:
    image = Image(PILImage.new("RGB", (20, 10)))
    resized = ImageTransformer.resize(image, width=10, height=5)
    assert resized.size == (10, 5)


def test_resize_with_aspect_ratio_shrinks_longer_side() -> None:
    image = Image(PILImage.new("RGB", (20, 10)))
    resized = ImageTransformer.resize_with_aspect_ratio(image, max_size=10)
    assert resized.size == (10, 5)


def test_resize_with_aspect_ratio_noop_when_already_small() -> None:
    image = Image(PILImage.new("RGB", (5, 5)))
    result = ImageTransformer.resize_with_aspect_ratio(image, max_size=10)
    assert result is image


def test_to_rgb_converts_mode() -> None:
    image = Image(PILImage.new("L", (4, 4)))
    rgb = ImageTransformer.to_rgb(image)
    assert rgb.channels == 3


def test_to_grayscale_converts_mode() -> None:
    image = Image(PILImage.new("RGB", (4, 4)))
    gray = ImageTransformer.to_grayscale(image)
    assert gray.channels == 1


def test_to_numpy_shape() -> None:
    image = Image(PILImage.new("RGB", (20, 10)))
    arr = ImageTransformer.to_numpy(image)
    assert arr.shape == (10, 20, 3)


def test_composed_resize_and_grayscale() -> None:
    x = Image(PILImage.new("RGB", (20, 10)))
    x = ImageTransformer.resize(x, 10, 5)
    x = ImageTransformer.to_grayscale(x)
    assert x.size == (10, 5)
    assert x.channels == 1
