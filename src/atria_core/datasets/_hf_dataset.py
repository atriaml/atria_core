from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import aiohttp
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets._dataset import (
    Dataset,
    DatasetConfig,
    T_BaseDataInstance,
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


@pydantic_dataclass(frozen=True)
class HuggingfaceDatasetConfig(DatasetConfig):
    config_name: str | None = None
    dataset_dir_name: str | None = None


T_HuggingfaceDatasetConfig = TypeVar(
    "T_HuggingfaceDatasetConfig", bound=HuggingfaceDatasetConfig
)


class HuggingfaceDataset(
    Dataset[T_HuggingfaceDatasetConfig, T_BaseDataInstance],
    Generic[T_HuggingfaceDatasetConfig, T_BaseDataInstance],
):
    __abstract__ = True

    def __init__(
        self,
        repo: str,
        config: T_HuggingfaceDatasetConfig,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
    ) -> None:
        self._repo = repo
        super().__init__(
            config, data_dir=data_dir, access_token=access_token, split=split
        )

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        """HF handles its own downloading internally during streaming --
        this just builds the data_dir-dependent state every other hook
        needs (builder, download manager, split generators), once, up
        front. _download runs first in Dataset._build_split_iterators, so
        by the time _metadata/_available_splits/_build_split_iterator run,
        this state is already there."""
        from datasets import load_dataset_builder

        self._builder: datasets.DatasetBuilder = load_dataset_builder(
            self._repo,
            name=self.config.config_name,
            cache_dir=data_dir,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
        )
        self._download_manager = self._prepare_download_manager(data_dir, access_token)
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
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
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
