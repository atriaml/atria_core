from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from atria_core.datasets._constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR
from atria_core.logger import get_logger
from atria_core.transforms.functional import image as image_functional
from atria_core.types import (
    BaseDataInstance,
    Image,
    ImageInstance,
    PdfPage,
    SinglePageDocumentInstance,
)

logger = get_logger(__name__)


def transform_hash(transform: Callable[[Any], Any] | None) -> str | None:
    """Storage layer doesn't know or care what a transform does -- it only
    needs a stable identity to fold into the cache path. Transforms that
    declare their own `.hash` (e.g. PreprocessTransform) use that; a
    Compose (from _split_iterators) is hashed recursively over its
    first/second so composing transforms doesn't collapse to a
    non-deterministic id(); arbitrary callables (e.g. a user's
    process_and_cache transform) fall back to a hash of their repr."""
    if transform is None:
        return None
    from atria_core.datasets._split_iterators import Compose

    if isinstance(transform, Compose):
        first_hash = transform_hash(transform.first) or "none"
        second_hash = transform_hash(transform.second) or "none"
        return hashlib.md5(f"{first_hash}|{second_hash}".encode()).hexdigest()[:8]
    declared_hash = getattr(transform, "hash", None)
    if declared_hash is not None:
        return str(declared_hash)
    return hashlib.md5(repr(transform).encode()).hexdigest()[:8]


class PreprocessTransform:
    """Flag-driven output transform applied per-sample before writing to
    storage. materialize_content=True makes to_dict() embed content as
    bytes instead of requiring a file path -- needed for msgpack/tar-shard
    storage, which bundles many small binary blobs into shard files rather
    than reading/writing them one file at a time."""

    def __init__(
        self,
        materialize_content: bool = True,
        resize_images: bool = False,
        image_max_size: int | tuple[int, int] | None = None,
    ) -> None:
        self._materialize_content = materialize_content
        self._resize_images = resize_images
        self._image_max_size = image_max_size

    @property
    def hash(self) -> str:
        return hashlib.md5(
            f"materialize_content:{self._materialize_content}|"
            f"resize_images:{self._resize_images}|"
            f"image_max_size:{self._image_max_size}".encode()
        ).hexdigest()[:8]

    def __call__(self, sample: BaseDataInstance) -> BaseDataInstance:
        if isinstance(sample, ImageInstance):
            processed = self._process_visual(sample.image)
            assert isinstance(processed, Image)
            return replace(sample, image=processed)
        if isinstance(sample, SinglePageDocumentInstance):
            return replace(sample, visual=self._process_visual(sample.visual))
        return sample

    def _process_visual(self, visual: Image | PdfPage) -> Image | PdfPage:
        if self._materialize_content:
            visual = visual.load()
        if self._resize_images and visual.content is not None:
            resized = self._resize(Image.from_source(visual.content))
            visual = replace(visual, content=resized.require_content())
        return visual

    def _resize(self, image: Image) -> Image:
        assert self._image_max_size is not None
        if isinstance(self._image_max_size, tuple):
            return image_functional.resize(
                image, width=self._image_max_size[0], height=self._image_max_size[1]
            )
        return image_functional.resize_with_aspect_ratio(
            image, max_size=self._image_max_size
        )


def _validate_data_dir(data_dir: str | Path) -> str:
    data_dir = Path(data_dir)
    if data_dir.exists():
        assert data_dir.is_dir(), (
            f"Data directory `{data_dir.absolute()}` exists but is not a directory."
        )
    else:
        logger.warning(
            f"Data directory `{data_dir.absolute()}` does not exist. Creating it."
        )
        data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def _default_data_dir(class_name: str) -> str:
    return str(_DEFAULT_ATRIA_DATASETS_CACHE_DIR / class_name)
