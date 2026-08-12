from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image as PILImage

from atria_core.types._generic._image import Image


def test_construct_from_pil_image_is_already_loaded(
    sample_image: PILImage.Image,
) -> None:
    image = Image.from_source(sample_image)
    assert image.file_path is None
    assert image.content is sample_image
    assert image.size == (16, 12)
    assert image.width == 16
    assert image.height == 12
    assert image.channels == 3
    assert image.shape == (3, 16, 12)


def test_construct_from_path_is_lazy(sample_image_path: Path) -> None:
    image = Image.from_source(sample_image_path)
    assert image.file_path == str(sample_image_path)
    assert image.content is None


def test_require_content_without_load_raises(sample_image_path: Path) -> None:
    image = Image.from_source(sample_image_path)
    with pytest.raises(AssertionError):
        image.require_content()


def test_load_from_path_populates_content(sample_image_path: Path) -> None:
    image = Image.from_source(sample_image_path)
    image = image.load()
    assert image.content is not None
    assert image.size == (16, 12)


def test_lazy_crop_survives_serialization(sample_image_path: Path) -> None:
    image = Image(file_path=str(sample_image_path), crop_box=(2, 3, 12, 10))

    restored = Image.from_dict(image.to_dict())

    assert restored.crop_box == (2, 3, 12, 10)
    assert restored.load().size == (10, 7)


def test_load_is_idempotent_when_already_loaded(sample_image: PILImage.Image) -> None:
    image = Image.from_source(sample_image)
    loaded = image.load()
    assert loaded is image
    assert loaded.content is sample_image


def test_to_dict_from_dict_roundtrip_via_file_path(sample_image_path: Path) -> None:
    image = Image.from_source(sample_image_path)
    data = image.to_dict()
    assert data == {"file_path": str(sample_image_path)}
    restored = Image.from_dict(data)
    assert restored.file_path == str(sample_image_path)
    assert restored.content is None


def test_to_dict_from_dict_roundtrip_via_embedded_bytes(
    sample_image: PILImage.Image,
) -> None:
    image = Image.from_source(sample_image)
    data = image.to_dict()
    assert "content_bytes" in data
    restored = Image.from_dict(data)
    assert restored.file_path is None
    assert restored.content is not None
    assert restored.size == image.size


def test_to_dict_without_path_or_content_raises() -> None:
    with pytest.raises(ValueError):
        Image().to_dict()


def test_equality_by_field() -> None:
    a = Image.from_source("/some/path.png")
    b = Image.from_source("/some/path.png")
    c = Image.from_source("/other/path.png")
    assert a == b
    assert a != c
