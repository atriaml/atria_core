from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import aiohttp
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets._constants import _HF_DOWNLOAD_TIMEOUT_SECONDS
from atria_core.datasets._dataset import (
    Dataset,
    DatasetConfig,
    T_DataInstance,
)
from atria_core.logger import get_logger
from atria_core.types import DatasetMetadata, DatasetSplitType

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


@pydantic_dataclass(frozen=True)
class HuggingfaceDatasetConfig(DatasetConfig):
    """Params for a dataset streamed from the Hugging Face hub."""

    config_name: str | None = None


T_HuggingfaceDatasetConfig = TypeVar(
    "T_HuggingfaceDatasetConfig",
    bound=HuggingfaceDatasetConfig,
    default=HuggingfaceDatasetConfig,
)


class HuggingfaceDataset(
    Dataset[T_DataInstance, T_HuggingfaceDatasetConfig],
    Generic[T_DataInstance, T_HuggingfaceDatasetConfig],
):
    """Streams a dataset straight off the Hugging Face hub, letting the
    `datasets` library handle downloading and caching internally."""

    __abstract__ = True
    def __init__(
        self,
        *,
        repo: str,
        config: T_HuggingfaceDatasetConfig | None = None,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
        dataset_dir_name: str | None = None,
    ) -> None:
        """Stream a dataset from the Hugging Face hub.

        Args:
            repo: Hub repo id, e.g. `"ylecun/mnist"`.
            config: Params for this dataset. Defaults to its generic config type.
            data_dir: Where Hugging Face caches its downloads.
            access_token: Credential for gated repos.
            split: Build only this split, instead of every available one.
        """
        self._repo = repo
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

        The `datasets` library fetches data itself while streaming, so nothing
        is downloaded here. This runs before any other hook, so the state it
        builds is available to metadata and split construction.
        """
        from datasets import load_dataset_builder

        self._builder: datasets.DatasetBuilder = load_dataset_builder(
            self._repo,
            name=self.config.config_name,
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
        if "packaged_modules" in str(self._builder.__module__):
            return datasets.DownloadManager(
                dataset_name=self.config.config_name,
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
        return self._builder._as_streaming_dataset_single(
            self._hf_split_generators[split]
        )
