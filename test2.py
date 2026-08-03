from atria_core.types._data_instance._image_instance import ImageInstance
from atria_core.types._generic._annotations import (
    AnnotationType,
    ClassificationAnnotation,
)

image = ImageInstance(sample_id="s1", image=None)
image.add_annotation(ClassificationAnnotation(label_value=0, label_name="label1"))
ann = image.get_annotation_by_type(AnnotationType.classification)
