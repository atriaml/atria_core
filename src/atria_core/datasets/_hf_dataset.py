from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import aiohttp

from atria_core.datasets._common import (
    T_BaseDataInstance,
    T_HuggingfaceDatasetConfig,
)
from atria_core.datasets._constants import _DEFAULT_DOWNLOAD_PATH
from atria_core.datasets._dataset import Dataset
from atria_core.logger import get_logger
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
)

if TYPE_CHECKING:
    import datasets

logger = get_logger(__name__)


class HuggingfaceDataset(
    Dataset[T_HuggingfaceDatasetConfig, T_BaseDataInstance],
    Generic[T_HuggingfaceDatasetConfig, T_BaseDataInstance],
):
    __abstract__ = True

    def __init__(self, config: T_HuggingfaceDatasetConfig) -> None:
        super().__init__(config)
        self.__hf_dataset_builder: datasets.DatasetBuilder | None = None
        self._hf_split_generators: dict[DatasetSplitType, Any] | None = None

    @property
    def _hf_dataset_builder(self) -> datasets.DatasetBuilder:
        if self.__hf_dataset_builder is None:
            from datasets import load_dataset_builder

            self.__hf_dataset_builder = load_dataset_builder(
                self.config.hf_repo,
                name=self.config.hf_config_name,
                cache_dir=self._data_dir,
                storage_options={
                    "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
                },
            )
        return self.__hf_dataset_builder

    def _custom_download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        return {}

    def _prepare_download_manager(
        self, data_dir: str, download_dir: str
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

    def _available_splits(self) -> list[DatasetSplitType]:
        if self._hf_split_generators is None:
            assert self._data_dir is not None, (
                "data_dir must be set before calling _available_splits(). "
                "Use load() or cache() to build the dataset."
            )
            download_dir = Path(self._data_dir) / _DEFAULT_DOWNLOAD_PATH
            download_dir.mkdir(parents=True, exist_ok=True)
            download_manager = self._prepare_download_manager(
                self._data_dir, download_dir=str(download_dir)
            )
            hf_split_map = {
                "train": DatasetSplitType.train,
                "validation": DatasetSplitType.validation,
                "test": DatasetSplitType.test,
            }
            self._hf_split_generators = {
                hf_split_map[sg.name]: sg
                for sg in self._hf_dataset_builder._split_generators(download_manager)
            }
        return list(self._hf_split_generators.keys())

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata.from_huggingface_info(self._hf_dataset_builder.info)

    def _split_iterator(self, split: DatasetSplitType, data_dir: str) -> Iterable[Any]:
        assert self._hf_split_generators is not None, (
            "Hugging Face split generators have not been initialized. "
            "Ensure that _available_splits() has been called."
        )
        return self._hf_dataset_builder._as_streaming_dataset_single(  # type: ignore[no-any-return]
            self._hf_split_generators[split]
        )


class HuggingfaceImageDataset(
    HuggingfaceDataset[T_HuggingfaceDatasetConfig, ImageInstance],
    Generic[T_HuggingfaceDatasetConfig],
):
    __abstract__ = True
    __data_model__ = ImageInstance


class HuggingfaceDocumentDataset(
    HuggingfaceDataset[T_HuggingfaceDatasetConfig, DocumentInstance],
    Generic[T_HuggingfaceDatasetConfig],
):
    __abstract__ = True
    __data_model__ = DocumentInstance
