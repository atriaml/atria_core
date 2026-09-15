"""Example: Creating and using ImageInstance (image classification/detection)."""

import numpy as np
from PIL import Image as PILImage

from atria_core.logger import get_logger
from atria_core.types import (
    AnnotatedObject,
    Image,
    ImageInstance,
    ObjectDetectionAnnotation,
)
from atria_core.types._generic._annotations import ClassificationAnnotation

logger = get_logger(__name__)


def main() -> None:
    # Create a simple image instance for classification
    pil_img = PILImage.new("RGB", (800, 600), color="blue")

    img_instance = ImageInstance(
        sample_id="img_001",
        image=Image(
            file_path="/data/images/sample.jpg",
            content=pil_img,
        ),
    ).add_annotation(ClassificationAnnotation(label_value=1, label_name="cat"))

    logger.info("Example Image Instance:\n%s", img_instance)

    # Create an image instance with object detection annotations
    detection_annotation = ObjectDetectionAnnotation.from_objects([
        AnnotatedObject(
            label_value=1,
            label_name="person",
            bbox=np.asarray([100.0, 200.0, 250.0, 500.0]),
        ),
        AnnotatedObject(
            label_value=2,
            label_name="car",
            bbox=np.asarray([400.0, 300.0, 800.0, 600.0]),
        ),
        AnnotatedObject(
            label_value=1,
            label_name="person",
            bbox=np.asarray([1200.0, 250.0, 1350.0, 550.0]),
        ),
    ])
    img_instance_with_det_annotations = ImageInstance(
        sample_id="img_002",
        image=Image(
            file_path="/data/images/street.jpg",
            content=PILImage.new("RGB", (1920, 1080)),
        ),
    ).add_annotation(detection_annotation)

    logger.info(
        "Example Image Instance with Object Annotations:\n%s",
        img_instance_with_det_annotations,
    )


if __name__ == "__main__":
    main()
