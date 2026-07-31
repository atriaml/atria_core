from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL.Image import Resampling


class Image:
    def __init__(self, source: str | Path | PILImage.Image) -> None:
        if isinstance(source, PILImage.Image):
            self.file_path = None
            self.content = source
        else:
            self.file_path = str(source)
            self.content = None

    def load(self):
        if self.content is None:
            from atria_core.types._utilities._image_encoding import _bytes_to_image
            from atria_core.types._utilities._url_fetchers import _load_bytes_from_uri

            self.content = _bytes_to_image(_load_bytes_from_uri(self.file_path))

    # -------------------------------------
    # Basic attributes
    # -------------------------------------
    @property
    def size(self) -> tuple[int, int]:
        return self.content.size

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def channels(self) -> int:
        return len(self.content.getbands())

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, *self.size)

    # -------------------------------------
    # Ops
    # -------------------------------------
    def to_numpy(self) -> np.ndarray:
        return np.array(self.content)

    def to_rgb(self) -> Image:
        return Image(self.content.convert("RGB"))

    def to_grayscale(self) -> Image:
        return Image(self.content.convert("L"))

    def resize(
        self, width: int, height: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        return Image(self.content.resize((width, height), resample))

    def resize_with_aspect_ratio(
        self, max_size: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        assert max_size > 0, "max_size must be > 0"

        if max(self.width, self.height) <= max_size:
            return self

        if self.width >= self.height:
            new_w = max_size
            new_h = int(self.height * (max_size / self.width))
        else:
            new_h = max_size
            new_w = int(self.width * (max_size / self.height))

        return self.resize(new_w, new_h, resample)

    # -------------------------------------
    # Dunder helpers
    # -------------------------------------
    def __repr__(self) -> str:
        return (
            f"Image(file_path={self.file_path!r}, width={self.width}, "
            f"height={self.height}, channels={self.channels})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return self.file_path == other.file_path and self.content == other.content
