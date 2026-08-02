from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_DOWNLOAD_PATH,
)
from atria_core.datasets._split_iterators import SplitIterator
from atria_core.logger import get_logger
from atria_core.transforms.functional import image as image_functional
from atria_core.types import (
    BaseDataInstance,
    DatasetSplitType,
    DocumentInstance,
    Image,
    ImageInstance,
)
from atria_core.types._generic._documents import SinglePageDocument

if TYPE_CHECKING:
    from atria_core.datasets._dataset import Dataset

logger = get_logger(__name__)


class ComposedTransform:
    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self._transforms = transforms

    def __call__(self, sample: BaseDataInstance) -> BaseDataInstance:
        for transform in self._transforms:
            sample = transform(sample)
        return sample


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

    def __call__(self, sample: BaseDataInstance) -> BaseDataInstance:
        if isinstance(sample, ImageInstance):
            return replace(sample, image=self._process_image(sample.image))
        if isinstance(sample, DocumentInstance) and isinstance(
            sample.document, SinglePageDocument
        ):
            return replace(sample, document=self._process_document(sample.document))
        return sample

    def _process_image(self, image: Image) -> Image:
        if self._materialize_content:
            image = image.load()
        if self._resize_images and image.content is not None:
            image = self._resize(image)
        return image

    def _process_document(self, document: SinglePageDocument) -> SinglePageDocument:
        if not self._resize_images:
            return document
        resized = self._resize(Image.from_source(document.image))
        return replace(document, image=resized.require_content())

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


def _default_data_dir(dataset: Dataset[Any, Any]) -> str:
    name = dataset.config.dataset_name or dataset.__class__.__name__
    return str(_DEFAULT_ATRIA_DATASETS_CACHE_DIR / name)


def _prepare_downloads(
    dataset: Dataset[Any, Any], data_dir: str, access_token: str | None
) -> dict[str, Path] | None:
    from atria_core.datasets._dataset import Dataset as _Dataset
    from atria_core.datasets._download._download_manager import AtriaDownloadManager

    if dataset.__requires_access_token__ and access_token is None:
        logger.warning(
            "access_token must be passed to download this dataset. "
            f"See `{dataset.metadata.homepage}` for instructions to get the access token"
        )

    if dataset._custom_download.__func__ is not _Dataset._custom_download:  # type: ignore[attr-defined]
        dataset._custom_download(data_dir, access_token)
        return None
    download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
    download_dir.mkdir(parents=True, exist_ok=True)
    download_manager = AtriaDownloadManager(
        data_dir=Path(data_dir), download_dir=download_dir
    )
    download_urls = dataset._download_urls()
    if not download_urls:
        return None
    downloaded = download_manager.download_and_extract(
        download_urls,
        extract=dataset.__extract_downloads__,
        access_token=access_token,
    )
    logger.info(f"Downloaded files {downloaded}")
    return downloaded


def _prepare_split(
    dataset: Dataset[Any, Any],
    split: DatasetSplitType,
    data_dir: str,
    split_iterator_type: type[SplitIterator[Any]],
    materialize_content: bool = True,
    resize_images: bool = False,
    image_max_size: int | None = None,
    user_transform: Callable[[Any], Any] | None = None,
    for_cache: bool = False,
    base_iterator: object | None = None,
) -> SplitIterator[Any]:
    """for_cache=True materializes content for shard/tar storage; either
    way, base_iterator lets a caller reuse a raw iterator a prior call
    already built instead of calling dataset._split_iterator(...) again."""
    limits = {
        DatasetSplitType.train: dataset.config.max_train_samples,
        DatasetSplitType.validation: dataset.config.max_validation_samples,
        DatasetSplitType.test: dataset.config.max_test_samples,
    }
    base_tf = PreprocessTransform(
        materialize_content=materialize_content if for_cache else True,
        resize_images=resize_images,
        image_max_size=image_max_size,
    )
    output_transform: Callable[[Any], Any] = base_tf
    if user_transform is not None:
        output_transform = ComposedTransform([base_tf, user_transform])
    return split_iterator_type(
        split=split,
        data_model=dataset.data_model,
        input_transform=dataset.input_transform,
        base_iterator=(
            base_iterator
            if base_iterator is not None
            else dataset._split_iterator(split, data_dir)  # type: ignore[arg-type]
        ),
        max_len=limits[split],
        output_transform=output_transform,
    )
