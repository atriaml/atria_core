from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage

from atria_core.types._data_instance._image_instance import ImageInstance
from atria_core.types._generic._image import Image
from atria_core.types._serialization._artifact_store import ArtifactStore
from atria_core.types._serialization._row_codec import RowCodec


def test_to_row_produces_expected_shape(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    instance = ImageInstance(sample_id="s1", image=Image.from_source(PILImage.new("RGB", (4, 4))))

    row = RowCodec.to_row(instance, store)

    assert row["sample_id"] == "s1"
    assert row["type"] == "image_instance"
    data = json.loads(row["data_json"])
    assert data["sample_id"] == "s1"
    assert data["image"]["file_path"] == str(tmp_path / "artifacts" / "s1.png")


def test_to_row_does_not_mutate_original_instance(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    image = Image.from_source(PILImage.new("RGB", (4, 4)))
    instance = ImageInstance(sample_id="s1", image=image)

    RowCodec.to_row(instance, store)

    # Original instance's Image is untouched -- to_row materializes a copy.
    assert instance.image is image
    assert instance.image.file_path is None


def test_from_row_reconstructs_instance(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    instance = ImageInstance(sample_id="s1", image=Image.from_source(PILImage.new("RGB", (4, 4))))
    row = RowCodec.to_row(instance, store)

    restored = RowCodec.from_row(row)

    assert isinstance(restored, ImageInstance)
    assert restored.sample_id == "s1"
    assert restored.image.file_path == str(tmp_path / "artifacts" / "s1.png")
