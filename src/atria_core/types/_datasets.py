from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets.info import DatasetInfo

from atria_core.logger import get_logger
from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._common import DatasetSplitType

logger = get_logger(__name__)


@dataclass(frozen=True, repr=False)
class SplitConfig(BaseDataModel):
    """The split type and additional keyword arguments for generating a dataset split."""

    split: DatasetSplitType
    gen_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split.value, "gen_kwargs": self.gen_kwargs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitConfig:
        return cls(
            split=DatasetSplitType(data["split"]), gen_kwargs=data.get("gen_kwargs", {})
        )


@dataclass(frozen=True, repr=False)
class DatasetShardInfo(BaseDataModel):
    """Information about a single dataset shard."""

    url: str = ""
    shard: int = 1
    nsamples: int = 0
    filesize: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "shard": self.shard,
            "nsamples": self.nsamples,
            "filesize": self.filesize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetShardInfo:
        return cls(
            url=data.get("url", ""),
            shard=data.get("shard", 1),
            nsamples=data.get("nsamples", 0),
            filesize=data.get("filesize", 0),
        )


@dataclass(frozen=True, repr=False)
class SplitInfo(BaseDataModel):
    """Aggregate information about a dataset split, across all its shards."""

    num_bytes: int
    num_examples: int
    shardlist: list[DatasetShardInfo]

    @classmethod
    def from_shard_info_list(cls, shard_list: list[DatasetShardInfo]) -> SplitInfo:
        num_bytes = sum(shard.filesize for shard in shard_list)
        num_examples = sum(shard.nsamples for shard in shard_list)
        return cls(num_bytes=num_bytes, num_examples=num_examples, shardlist=shard_list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_bytes": self.num_bytes,
            "num_examples": self.num_examples,
            "shardlist": [shard.to_dict() for shard in self.shardlist],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitInfo:
        return cls(
            num_bytes=data["num_bytes"],
            num_examples=data["num_examples"],
            shardlist=[DatasetShardInfo.from_dict(s) for s in data["shardlist"]],
        )

    def to_file(self, file_path: Path | str) -> None:
        with open(str(file_path), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_file(cls, file_path: Path | str) -> SplitInfo:
        with Path(file_path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True, repr=False)
class DatasetLabels(BaseDataModel):
    """Classification and token labels for a dataset."""

    classification: list[str] | None = None
    ser: list[str] | None = None
    layout: list[str] | None = None

    @classmethod
    def _infer_from_huggingface_features(cls, features: Any) -> DatasetLabels:
        import datasets

        instance_labels = None
        object_labels = None
        token_labels = None
        for key, value in features.items():
            if isinstance(value, datasets.ClassLabel):
                instance_labels = value.names
            elif isinstance(value, datasets.Sequence) and isinstance(
                value.feature, datasets.ClassLabel
            ):
                token_labels = value.feature.names
            elif isinstance(value, list) and "objects" in key:
                for _, obj_value in value[0].items():
                    if isinstance(obj_value, datasets.ClassLabel):
                        object_labels = obj_value.names
        if instance_labels is None and object_labels is None and token_labels is None:
            logger.warning(
                "No labels found in the dataset features. Please check the dataset structure."
            )
        return cls(
            classification=instance_labels, layout=object_labels, ser=token_labels
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "ser": self.ser,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetLabels:
        return cls(
            classification=data.get("classification"),
            ser=data.get("ser"),
            layout=data.get("layout"),
        )


@dataclass(frozen=True, repr=False)
class DatasetMetadata(BaseDataModel):
    """Metadata for a dataset, including configuration and labels."""

    homepage: str | None = None
    description: str | None = None
    license: str | None = None
    citation: str | None = None
    dataset_labels: DatasetLabels = field(default_factory=DatasetLabels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "homepage": self.homepage,
            "description": self.description,
            "license": self.license,
            "citation": self.citation,
            "dataset_labels": self.dataset_labels.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        dataset_labels = data.get("dataset_labels")
        return cls(
            homepage=data.get("homepage"),
            description=data.get("description"),
            license=data.get("license"),
            citation=data.get("citation"),
            dataset_labels=DatasetLabels.from_dict(dataset_labels)
            if dataset_labels is not None
            else DatasetLabels(),
        )

    def to_file(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_file(cls, file_path: str) -> DatasetMetadata:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset info file not found at {file_path}")
        with open(file_path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_huggingface_info(cls, info: DatasetInfo) -> DatasetMetadata:
        return cls(
            citation=info.citation,
            homepage=info.homepage,
            license=info.license,
            dataset_labels=DatasetLabels._infer_from_huggingface_features(
                info.features
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        restored = DatasetMetadata.from_dict(state_dict)
        self.__dict__.update(restored.__dict__)


@dataclass(frozen=True, repr=False)
class DatasetStorageInfo(BaseDataModel):
    """Storage information for a dataset: its metadata and split info."""

    metadata: DatasetMetadata
    split_info: SplitInfo

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "split_info": self.split_info.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetStorageInfo:
        return cls(
            metadata=DatasetMetadata.from_dict(data["metadata"]),
            split_info=SplitInfo.from_dict(data["split_info"]),
        )

    def to_file(self, file_path: Path | str) -> None:
        with open(str(file_path), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_file(cls, file_path: Path | str) -> DatasetStorageInfo:
        with Path(file_path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
