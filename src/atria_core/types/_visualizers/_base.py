from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from PIL.Image import Image as PILImage

from atria_core.logger import get_logger
from atria_core.types._data_instance._base import AnnotationNotFoundError
from atria_core.types._generic._annotations import AnnotationType

if TYPE_CHECKING:
    from atria_core.types._data_instance._base import BaseDataInstance

logger = get_logger(__name__)


class Visualizer:
    def __init__(self, instance: BaseDataInstance) -> None:
        self.instance = instance

    @property
    def output_name(self) -> str:
        try:
            classification_annotation = self.instance.get_annotation_by_type(
                AnnotationType.classification
            )
            return f"{self.instance.sample_id}_label={classification_annotation.label_name}"
        except AnnotationNotFoundError:
            return self.instance.sample_id

    def _load_image(self) -> PILImage:
        from atria_core.types._data_instance._document_instance import DocumentInstance
        from atria_core.types._data_instance._image_instance import ImageInstance
        from atria_core.types._generic._documents import MultiPageDocument

        if isinstance(self.instance, ImageInstance):
            self.instance.image.load()
            return cast(PILImage, self.instance.image.content)

        if isinstance(self.instance, DocumentInstance):
            document = self.instance.document
            if isinstance(document, MultiPageDocument):
                raise TypeError(
                    "Visualizer requires a SinglePageDocument; got a MultiPageDocument."
                )
            return document.image

        raise TypeError(
            "Visualizer can only load images for ImageInstance or DocumentInstance types."
        )

    def _draw_on_image(self, image: PILImage) -> PILImage:
        return image

    def visualize(self, output_path: str) -> PILImage:
        image = self._load_image()
        image = self._draw_on_image(image.copy().convert("RGB"))
        Path(output_path).mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Saving visualization for sample {self.instance.sample_id} to {output_path}"
        )
        image.save(Path(output_path) / f"{self.output_name}.png")
        return image
