from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import computed_field
from rich.repr import RichReprResult

from atria_core.types._base._data_model import BaseDataModel
from atria_core.types._pydantic import ValidatedPILImage

if TYPE_CHECKING:
    from atria_core.types._generic._ops._image_ops import ImageOps


class Image(BaseDataModel):
    content: ValidatedPILImage

    @computed_field
    @property
    def size(self) -> tuple[int, int]:
        assert self.content is not None, "Image content is not loaded."
        return (self.content.size[0], self.content.size[1])

    @computed_field
    @property
    def width(self) -> int:
        return self.size[0]

    @computed_field
    @property
    def height(self) -> int:
        return self.size[1]

    @computed_field
    @property
    def channels(self) -> int:
        assert self.content is not None
        return len(self.content.getbands())

    @computed_field
    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, *self.size)

    @property
    def ops(self) -> ImageOps:
        from ._ops._image_ops import ImageOps

        return ImageOps(self)

    @classmethod
    def from_file_path_or_uri(cls, file_uri: str) -> Self:
        from atria_core.types._utilities._image_encoding import _bytes_to_image
        from atria_core.types._utilities._url_fetchers import ResourceLoader

        content = _bytes_to_image(ResourceLoader.for_uri(file_uri).load_bytes())
        return cls(content=content)

    def __rich_repr__(self) -> RichReprResult:  # type: ignore
        yield from super().__rich_repr__()
        if self.content is not None:
            yield "width", self.width
            yield "height", self.height
            yield "channels", self.channels
