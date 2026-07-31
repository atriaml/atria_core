from functools import cached_property

from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._generic._image import Image
from atria_core.types._pydantic import StrField


class ImageInstance(BaseDataInstance):
    file_path: StrField

    @cached_property
    def image(self) -> Image:
        return Image.from_file_path_or_uri(self.file_path)
