from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets._cacher import Cacher, FileStorageType
from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import SinglePageDocumentInstance
from atria_core.types._generic._annotations import TranscriptionAnnotation
from atria_core.types._generic._image import Image

_REPO = "fhswf/german_handwriting"


@pydantic_dataclass(frozen=True)
class FHSWFGermanHandwritingConfig(HuggingfaceDatasetConfig):
    config_name: str = "default"


class InputTransform:
    def __call__(self, sample: dict[str, Any]) -> SinglePageDocumentInstance:
        return SinglePageDocumentInstance(
            sample_id=str(uuid.uuid4()), visual=Image(content=sample["image"])
        ).add_annotation(TranscriptionAnnotation(text=sample["text"]))


class FHSWFGermanHandwriting(
    HuggingfaceDataset[SinglePageDocumentInstance, FHSWFGermanHandwritingConfig]
):
    def __init__(
        self, *, config: FHSWFGermanHandwritingConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(repo=_REPO, config=config, **kwargs)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()


def fhswf_german_handwriting(
    config_name: str = "default", **kwargs: Any
) -> FHSWFGermanHandwriting:
    """Build the FHSWF German handwriting dataset."""
    return FHSWFGermanHandwriting(
        config=FHSWFGermanHandwritingConfig(config_name=config_name), **kwargs
    )


def main() -> None:
    dataset = fhswf_german_handwriting()
    Cacher(FileStorageType.MSGPACK).cache(dataset)
    for sample in dataset.train:
        print(sample)


if __name__ == "__main__":
    main()
