from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL.Image import Image as PILImage

from atria_core.logger import get_logger
from atria_core.types._generic._annotations import AnnotationType
from atria_core.types._generic._documents import MultiPageDocument
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._utilities._viz import _draw_bboxes_on_image
from atria_core.types._visualizers._base import Visualizer

if TYPE_CHECKING:
    from atria_core.types._data_instance._document_instance import DocumentInstance
logger = get_logger(__name__)


class DocumentVisualizer(Visualizer):
    def __init__(self, instance: DocumentInstance) -> None:
        self.instance: DocumentInstance = instance

    def _draw_on_image(
        self,
        image: PILImage,
        draw_segment_bboxes: bool = False,
        draw_word_labels: bool = False,
    ) -> PILImage:
        document = self.instance.document
        if isinstance(document, MultiPageDocument):
            raise TypeError(
                "DocumentVisualizer requires a SinglePageDocument; got a MultiPageDocument."
            )

        content = document.content
        if content is None or content.elements is None:
            return image

        words = content.elements.at(OCRLevel.word)
        if words.bboxes is None or len(words) == 0:
            return image

        # ElementArray.bboxes is always normalized to [0, 1]; scale to pixels
        # for drawing.
        scale = np.array([image.width, image.height, image.width, image.height])
        bboxes_arr = (
            content.elements.segment_bboxes(OCRLevel.word)
            if draw_segment_bboxes
            else words.bboxes
        ) * scale

        bbox_labels = None
        if draw_word_labels:
            try:
                ann = self.instance.get_annotation_by_type(
                    annotation_type=AnnotationType.entity_labeling
                )
                bbox_labels = ann.label_names
            except Exception:  # noqa: E722
                pass

        # Draw bounding boxes on the image
        image = _draw_bboxes_on_image(
            image=image,
            bboxes=list(bboxes_arr),
            bboxes_text=words.texts.tolist() if words.texts is not None else None,
            bbox_labels=bbox_labels,
        )

        return image

    def visualize(
        self,
        output_path: str,
        draw_segment_bboxes: bool = False,
        draw_word_labels: bool = True,
    ) -> PILImage:
        image = self._load_image()
        image = self._draw_on_image(
            image.copy().convert("RGB"),
            draw_segment_bboxes=draw_segment_bboxes,
            draw_word_labels=draw_word_labels,
        )
        Path(output_path).mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Saving visualization for sample {self.instance.sample_id} to {output_path}"
        )
        image.save(Path(output_path) / f"{self.output_name}.png")
        return image
