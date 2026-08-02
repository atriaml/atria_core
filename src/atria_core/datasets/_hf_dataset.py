from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import aiohttp
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_core.datasets._constants import _DEFAULT_DOWNLOAD_PATH
from atria_core.datasets._dataset import Dataset, DatasetConfig, T_BaseDataInstance
from atria_core.datasets._split_iterators import IterableSplitIterator, SplitIterator
from atria_core.logger import get_logger
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
)

if TYPE_CHECKING:
    import datasets

logger = get_logger(__name__)


@pydantic_dataclass(frozen=True)
class HuggingfaceDatasetConfig(DatasetConfig):
    hf_repo: str = ""
    hf_config_name: str = ""


class HFSplitIterator(IterableSplitIterator[T_BaseDataInstance]):
    def __init__(self, hf_dataset: Iterable[Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._hf_dataset = hf_dataset

    def __iter__(self) -> Iterator[Any]:
        return iter(self._hf_dataset)


class HuggingfaceDataset(
    Dataset[HuggingfaceDatasetConfig, T_BaseDataInstance],
    Generic[T_BaseDataInstance],
):
    __abstract__ = True

    def __init__(
        self,
        config: HuggingfaceDatasetConfig,
        *,
        data_dir: str | None = None,
        access_token: str | None = None,
        split: DatasetSplitType | None = None,
    ) -> None:
        self._builder = self._prepare_builder(data_dir)
        self._download_manager = self._prepare_download_manager(
            data_dir=data_dir,
            download_dir=_DEFAULT_DOWNLOAD_PATH,
            access_token=access_token,
        )

        super().__init__(
            config,
            data_dir=data_dir,
            access_token=access_token,
            split=split,
        )

    def _prepare_builder(self, data_dir: str) -> datasets.DatasetBuilder:
        from datasets import load_dataset_builder

        return load_dataset_builder(
            self.config.hf_repo,
            name=self.config.hf_config_name,
            cache_dir=data_dir,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
        )

    def _prepare_download_manager(
        self, data_dir: str, download_dir: str, access_token: str | None = None
    ) -> datasets.DownloadManager:
        import datasets

        download_config = datasets.DownloadConfig(
            cache_dir=download_dir,
            force_download=False,
            force_extract=False,
            use_etag=False,
            delete_extracted=False,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
            token=access_token,
        )
        if "packaged_modules" in str(self._hf_dataset_builder.__module__):
            return datasets.DownloadManager(
                dataset_name=self.config.hf_config_name,
                data_dir=data_dir,
                download_config=download_config,
                record_checksums=False,
            )
        return datasets.DownloadManager(
            data_dir=data_dir, download_config=download_config, record_checksums=False
        )

    def _available_splits(
        self, _: str
    ) -> list[DatasetSplitType, datasets.SplitGenerator]:
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata.from_huggingface_info(self._hf_dataset_builder.info)

    def _build_split_iterator(
        self,
        split: DatasetSplitType,
        data_dir: str,
        access_token: str | None = None,
    ) -> SplitIterator[T_BaseDataInstance]:
        hf_dataset = self._hf_dataset_builder._split_generators(
            self._download_manager
        )._as_streaming_dataset_single(self._hf_split_generators[split.value])
        return HFSplitIterator(
            hf_dataset=hf_dataset,
            split=split,
            data_model=self.data_model,
        )

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        pass
