from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from atria_core.types._base_data_model import BaseDataModel


@dataclass(init=False, repr=False)
class Image(BaseDataModel):
    file_path: str | None = None
    content: PILImage.Image | None = None

    def __init__(self, source: str | Path | PILImage.Image) -> None:
        if isinstance(source, PILImage.Image):
            self.file_path = None
            self.content = source
        else:
            self.file_path = str(source)
            self.content = None

    def load(self) -> None:
        if self.content is None:
            from atria_core.types._utilities._image_encoding import _bytes_to_image
            from atria_core.types._utilities._url_fetchers import ResourceLoader

            assert self.file_path is not None, "Image has neither content nor file_path"
            loader = ResourceLoader.for_uri(self.file_path)
            self.content = _bytes_to_image(loader.load_bytes())

    def require_content(self) -> PILImage.Image:
        """Returns the loaded PIL image, or raises if load() hasn't been called."""
        assert self.content is not None, "Image content is not loaded; call load() first"
        return self.content

    # -------------------------------------
    # Read-only derived views
    # -------------------------------------
    @property
    def size(self) -> tuple[int, int]:
        return self.require_content().size

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def channels(self) -> int:
        return len(self.require_content().getbands())

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, *self.size)

    def to_dict(self) -> dict[str, Any]:
        if self.file_path is None:
            raise ValueError(
                "Image must be file-backed before to_dict() -- materialize "
                "in-memory content to a file first (see ArtifactStore)."
            )
        return {"file_path": self.file_path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Image:
        return cls(data["file_path"])
