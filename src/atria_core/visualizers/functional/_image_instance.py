from __future__ import annotations

from pathlib import Path

import numpy as np

from atria_core.logger import get_logger
from atria_core.transforms import functional as F
from atria_core.types import AnnotationType, ImageInstance
from atria_core.visualizers._drawers._image import ImageDrawer
from atria_core.visualizers._drawers._style import DEFAULT_STYLE, DrawStyle
from atria_core.visualizers.functional._visualize import output_name

logger = get_logger(__name__)


def visualize_image_instance(
    instance: ImageInstance, output_dir: str, *, style: DrawStyle = DEFAULT_STYLE
) -> Path:
    image = instance.image.load().require_content().copy().convert("RGB")

    detection = instance.get_annotation_by_type(AnnotationType.object_detection)
    if detection is not None and detection.bboxes is not None:
        detection = F.bbox.unnormalize(detection, image.width, image.height)
        objects = detection.to_objects()
        labels = [o.label_name for o in objects if o.label_name is not None]
        ImageDrawer().draw(
            image, np.stack([o.bbox for o in objects]), labels=labels, style=style
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{output_name(instance)}.png"
    logger.debug(f"Saving visualization for sample {instance.sample_id} to {path}")
    image.save(path)
    return path
