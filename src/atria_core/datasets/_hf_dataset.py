from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import aiohttp

from atria_core.datasets._constants import _HF_DOWNLOAD_TIMEOUT_SECONDS
from atria_core.datasets._dataset import Dataset, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import DatasetMetadata, DatasetSplitType
from atria_core.types._data_instance._base import DataInstance

if TYPE_CHECKING:
    import datasets

logger = get_logger(__name__)

_HF_SPLIT_MAP = {
    "train": DatasetSplitType.train,
    "validation": DatasetSplitType.validation,
    "test": DatasetSplitType.test,
}


def _hf_storage_options() -> dict[str, Any]:
    """Return the fsspec client options used for every hub request."""
    return {
        "client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=_HF_DOWNLOAD_TIMEOUT_SECONDS)
        }
    }


_DEFAULT_SHUFFLE_SEED = 42
_DEFAULT_SHUFFLE_BUFFER_SIZE = 10_000


class HuggingfaceDatasetConfig(DatasetConfig):
    """Params for a dataset loaded from the Hugging Face hub."""

    shuffle: bool = False
    shuffle_seed: int = _DEFAULT_SHUFFLE_SEED
    shuffle_buffer_size: int = _DEFAULT_SHUFFLE_BUFFER_SIZE


class HuggingfaceDataset[
    T_DataInstance: DataInstance,
    T_HuggingfaceDatasetConfig: HuggingfaceDatasetConfig = HuggingfaceDatasetConfig,
](
    Dataset[T_DataInstance, T_HuggingfaceDatasetConfig],
):
    """Streams a dataset straight off the Hugging Face hub, letting the
    `datasets` library handle downloading and caching internally.

    A concrete subclass declares which hub repo (and, for repos with
    multiple named configs, which one) it streams via `__hf_repo__` and
    `__hf_config_name__` -- these identify the dataset itself, so they are
    fixed on the class rather than supplied by callers, the same way
    `__module_name__` is.
    """

    __abstract__ = True
    __hf_repo__: ClassVar[str]
    __hf_config_name__: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__abstract__ and "__hf_repo__" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} is not abstract and must define a "
                f"'__hf_repo__' class variable."
            )

    def __init__(
        self,
        *,
        config: T_HuggingfaceDatasetConfig | None = None,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        dataset_dir_name: str | None = None,
        streaming: bool = False,
    ) -> None:
        """Load a dataset from the Hugging Face hub.

        Args:
            config: Params for this dataset. Defaults to its generic config type.
            data_dir: Where Hugging Face caches its downloads.
            access_token: Credential for gated repos.
            split: Build only this split, instead of every available one.
            streaming: Whether to stream rows instead of materializing the
                dataset to disk first. Streaming avoids downloading/caching
                the whole dataset -- worth it for datasets too large to
                materialize. It comes at a real cost though: HF `datasets`'
                streaming reader does not always yield rows in the same order
                as the materialized (non-streaming) dataset -- confirmed
                directly for OpenAssistant/oasst1, where the two orders
                diverge after the first few rows, changing which
                conversations end up sampled downstream even with an
                identical shuffle seed. Defaults to `False` (materialize
                first) so a dataset's row order is the canonical one every
                other loader (e.g. plain `datasets.load_dataset(...,
                streaming=False)`) also sees; set `True` only when streaming
                is actually necessary.
        """
        self._streaming = streaming
        super().__init__(
            config=config,
            data_dir=data_dir,
            access_token=access_token,
            split=split,
            dataset_dir_name=dataset_dir_name,
        )

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        """Prepare the hub builder, download manager and split generators.

        When streaming, `_split_generators` runs against a
        `StreamingDownloadManager` -- the same one plain `datasets.load_dataset(...,
        streaming=True)` uses -- so resolving the split list only yields lazy
        URLs instead of eagerly downloading every shard. When not streaming, this
        also materializes the dataset to disk (`download_and_prepare`), which
        `_build_split_iterator` then reads via `as_dataset` -- the same path
        plain `datasets.load_dataset(..., streaming=False)` takes, so row
        order matches it. This runs before any other hook, so the state it
        builds is available to metadata and split construction.
        """
        from datasets import load_dataset_builder

        self._builder: datasets.DatasetBuilder = load_dataset_builder(
            type(self).__hf_repo__,
            name=type(self).__hf_config_name__,
            cache_dir=data_dir,
            storage_options=_hf_storage_options(),
        )
        self._download_manager = self._prepare_download_manager(
            data_dir=data_dir, access_token=access_token
        )
        self._hf_split_generators = {
            _HF_SPLIT_MAP[sg.name]: sg
            for sg in self._builder._split_generators(self._download_manager)
            if sg.name in _HF_SPLIT_MAP
        }
        if not self._streaming:
            self._builder.download_and_prepare()
        return {}

    def _prepare_download_manager(
        self, data_dir: str, access_token: str | None
    ) -> datasets.DownloadManager:
        import datasets

        download_config = datasets.DownloadConfig(
            cache_dir=data_dir,
            force_download=False,
            force_extract=False,
            use_etag=False,
            delete_extracted=False,
            storage_options=_hf_storage_options(),
            token=access_token,
        )
        if self._streaming:
            return datasets.StreamingDownloadManager(
                base_path=self._builder.base_path,
                download_config=download_config,
                dataset_name=type(self).__hf_config_name__,
                data_dir=data_dir,
            )
        if "packaged_modules" in str(self._builder.__module__):
            return datasets.DownloadManager(
                dataset_name=type(self).__hf_config_name__,
                data_dir=data_dir,
                download_config=download_config,
                record_checksums=False,
            )
        return datasets.DownloadManager(
            data_dir=data_dir, download_config=download_config, record_checksums=False
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(self._hf_split_generators.keys())

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata.from_huggingface_info(self._builder.info)

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> Any:
        if self._streaming:
            streamed_split = self._builder._as_streaming_dataset_single(
                self._hf_split_generators[split]
            )
            if self.config.shuffle:
                return streamed_split.shuffle(
                    seed=self.config.shuffle_seed,
                    buffer_size=self.config.shuffle_buffer_size,
                )
            return streamed_split
        hf_split_name = next(
            name for name, mapped in _HF_SPLIT_MAP.items() if mapped == split
        )
        return self._builder.as_dataset(split=hf_split_name)
