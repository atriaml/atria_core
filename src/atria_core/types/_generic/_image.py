from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from atria_core.types._base_data_model import BaseDataModel


@dataclass(frozen=True, repr=False)
class Image(BaseDataModel):
    file_path: str | None = None
    content: PILImage.Image | None = None

    @classmethod
    def from_source(cls, source: str | Path | PILImage.Image) -> Image:
        if isinstance(source, PILImage.Image):
            return cls(file_path=None, content=source)
        return cls(file_path=str(source), content=None)

    def load(self) -> Image:
        """Returns an Image with `content` populated -- `self` if already
        loaded, otherwise a new instance (content is never fetched in
        place)."""
        if self.content is not None:
            return self

        from atria_core.types._utilities._image_encoding import _bytes_to_image
        from atria_core.types._utilities._url_fetchers import ResourceLoader

        assert self.file_path is not None, "Image has neither content nor file_path"
        loader = ResourceLoader.for_uri(self.file_path)
        return replace(self, content=_bytes_to_image(loader.load_bytes()))

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
        return cls(file_path=data["file_path"])
