from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from atria_core.serialization import ArtifactStore
from atria_core.types import Image


def test_materialize_image_writes_file_and_returns_unloaded_image(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    original = PILImage.new("RGB", (5, 5), color="red")
    result = store.materialize_image("s1", Image.from_source(original))

    path = tmp_path / "artifacts" / "s1.png"
    assert path.exists()
    assert result.file_path == str(path)
    assert result.content is None

    result = result.load()
    assert np.array_equal(np.array(original), np.array(result.require_content()))


def test_materialize_pdf_copies_bytes(tmp_path: Path, sample_pdf_path: Path) -> None:
    store = ArtifactStore(tmp_path / "dataset")
    path = store.materialize_pdf("m1", str(sample_pdf_path))

    assert Path(path).exists()
    assert Path(path).read_bytes() == sample_pdf_path.read_bytes()
