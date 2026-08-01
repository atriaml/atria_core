from __future__ import annotations

from PIL import Image as PILImage

from atria_core.types._data_instance._image_instance import ImageInstance
from atria_core.types._generic._image import Image
from tests.types.builders import make_classification_annotation


def test_construct_and_repr() -> None:
    image = Image.from_source(PILImage.new("RGB", (4, 4)))
    instance = ImageInstance(sample_id="s1", image=image)
    assert repr(instance)
    assert instance.image is image


def test_equality() -> None:
    image = Image.from_source(PILImage.new("RGB", (4, 4)))
    a = ImageInstance(sample_id="s1", image=image)
    b = ImageInstance(sample_id="s1", image=image)
    c = ImageInstance(sample_id="s2", image=image)
    assert a == b
    assert a != c


def test_inherits_base_data_instance_behavior() -> None:
    ann = make_classification_annotation()
    image = Image.from_source(PILImage.new("RGB", (4, 4)))
    instance = ImageInstance(sample_id="s1", image=image).add_annotation(ann)
    assert instance.key == "s1"
    from atria_core.types._generic._annotations import AnnotationType

    assert instance.get_annotation_by_type(AnnotationType.classification) is ann
