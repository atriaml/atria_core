from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image as PILImage

from atria_core.types._generic._image import Image


def test_construct_from_pil_image_is_already_loaded(sample_image: PILImage.Image) -> None:
    image = Image(sample_image)
    assert image.file_path is None
    assert image.content is sample_image
    assert image.size == (16, 12)
    assert image.width == 16
    assert image.height == 12
    assert image.channels == 3
    assert image.shape == (3, 16, 12)


def test_construct_from_path_is_lazy(sample_image_path: Path) -> None:
    image = Image(sample_image_path)
    assert image.file_path == str(sample_image_path)
    assert image.content is None


def test_require_content_without_load_raises(sample_image_path: Path) -> None:
    image = Image(sample_image_path)
    with pytest.raises(AssertionError):
        image.require_content()


def test_load_from_path_populates_content(sample_image_path: Path) -> None:
    image = Image(sample_image_path)
    image.load()
    assert image.content is not None
    assert image.size == (16, 12)


def test_load_is_idempotent_when_already_loaded(sample_image: PILImage.Image) -> None:
    image = Image(sample_image)
    image.load()
    assert image.content is sample_image


def test_equality_by_field() -> None:
    a = Image("/some/path.png")
    b = Image("/some/path.png")
    c = Image("/other/path.png")
    assert a == b
    assert a != c
