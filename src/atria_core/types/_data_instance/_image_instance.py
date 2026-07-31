from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._generic._image import Image


class ImageInstance(BaseDataInstance):  # type: ignore[misc]
    image: Image
