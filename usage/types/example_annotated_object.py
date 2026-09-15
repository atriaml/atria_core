"""Example: Creating and using AnnotatedObject (object detection)."""

import numpy as np

from atria_core.logger import get_logger
from atria_core.types import AnnotatedObject

logger = get_logger(__name__)


def main() -> None:
    # Create an annotated object (e.g., detected person)
    person = AnnotatedObject(
        label_value=1,
        label_name="person",
        bbox=np.asarray([100.0, 150.0, 300.0, 400.0]),
        iscrowd=False,
    )

    # Log details
    logger.info("Example Annotation:\n%s", person)

    # Create multiple annotated objects
    annotated_objects = [
        AnnotatedObject(
            label_value=1,
            label_name="person",
            bbox=np.asarray([50.0, 60.0, 150.0, 300.0]),
        ),
        AnnotatedObject(
            label_value=2,
            label_name="car",
            bbox=np.asarray([200.0, 250.0, 450.0, 400.0]),
        ),
        AnnotatedObject(
            label_value=1,
            label_name="person",
            bbox=np.asarray([500.0, 100.0, 600.0, 350.0]),
            segmentation=np.asarray([
                [500.0, 100.0],
                [600.0, 100.0],
                [600.0, 350.0],
                [500.0, 350.0],
            ]),
        ),
    ]

    logger.info("All annotations:\n%s", annotated_objects)

    # Convert to dict
    person_dict = person.to_dict()
    logger.info("Serialized: %s", person_dict)

    # Round-trip
    restored = AnnotatedObject.from_dict(person_dict)
    logger.info("Restored: %s", restored)


if __name__ == "__main__":
    main()
