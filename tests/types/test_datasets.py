from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from atria_core.types._common import DatasetSplitType
from atria_core.types._datasets import (
    DatasetLabels,
    DatasetMetadata,
    DatasetShardInfo,
    DatasetStorageInfo,
    SplitConfig,
    SplitInfo,
)


class FakeClassLabel:
    def __init__(self, names):
        self.names = names


class FakeSequence:
    def __init__(self, feature):
        self.feature = feature


@pytest.fixture
def fake_huggingface_datasets():
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.ClassLabel = FakeClassLabel
    fake_datasets.Sequence = FakeSequence
    sys.modules["datasets"] = fake_datasets
    try:
        yield fake_datasets
    finally:
        del sys.modules["datasets"]


def make_shard(**overrides) -> DatasetShardInfo:
    kwargs = {"url": "s3://bucket/shard-0.tar", "shard": 0, "nsamples": 10, "filesize": 100}
    kwargs.update(overrides)
    return DatasetShardInfo(**kwargs)


def test_split_config_to_dict_from_dict_roundtrip() -> None:
    config = SplitConfig(split=DatasetSplitType.train, gen_kwargs={"seed": 42})
    data = config.to_dict()
    restored = SplitConfig.from_dict(data)
    assert restored == config


def test_dataset_shard_info_defaults() -> None:
    shard = DatasetShardInfo()
    assert shard.url == ""
    assert shard.shard == 1
    assert shard.nsamples == 0
    assert shard.filesize == 0


def test_dataset_shard_info_to_dict_from_dict_roundtrip() -> None:
    shard = make_shard()
    restored = DatasetShardInfo.from_dict(shard.to_dict())
    assert restored == shard


def test_split_info_from_shard_info_list_aggregates() -> None:
    shards = [make_shard(nsamples=10, filesize=100), make_shard(nsamples=5, filesize=50)]
    split_info = SplitInfo.from_shard_info_list(shards)
    assert split_info.num_bytes == 150
    assert split_info.num_examples == 15
    assert split_info.shardlist == shards


def test_split_info_to_file_from_file_roundtrip(tmp_path: Path) -> None:
    split_info = SplitInfo.from_shard_info_list([make_shard()])
    path = tmp_path / "split_info.json"
    split_info.to_file(path)
    restored = SplitInfo.from_file(path)
    assert restored == split_info


def test_dataset_labels_to_dict_from_dict_roundtrip() -> None:
    labels = DatasetLabels(classification=["cat", "dog"], ser=["O", "B-ENT"])
    restored = DatasetLabels.from_dict(labels.to_dict())
    assert restored == labels


def test_dataset_labels_infer_from_huggingface_features(fake_huggingface_datasets) -> None:
    features = {"label": FakeClassLabel(["cat", "dog"])}
    labels = DatasetLabels._infer_from_huggingface_features(features)
    assert labels.classification == ["cat", "dog"]


def test_dataset_metadata_to_file_from_file_roundtrip(tmp_path: Path) -> None:
    metadata = DatasetMetadata(
        homepage="https://example.com",
        description="a dataset",
        dataset_labels=DatasetLabels(classification=["cat", "dog"]),
    )
    path = tmp_path / "metadata.json"
    metadata.to_file(str(path))
    restored = DatasetMetadata.from_file(str(path))
    assert restored == metadata


def test_dataset_metadata_from_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DatasetMetadata.from_file(str(tmp_path / "missing.json"))


def test_dataset_metadata_state_dict_roundtrip() -> None:
    metadata = DatasetMetadata(homepage="https://example.com")
    state = metadata.state_dict()

    other = DatasetMetadata()
    other.load_state_dict(state)
    assert other == metadata


def test_dataset_metadata_from_huggingface_info(fake_huggingface_datasets) -> None:
    class FakeDatasetInfo:
        citation = "some citation"
        homepage = "https://example.com"
        license = "MIT"
        features: dict = {}

    metadata = DatasetMetadata.from_huggingface_info(FakeDatasetInfo())
    assert metadata.citation == "some citation"
    assert metadata.homepage == "https://example.com"
    assert metadata.license == "MIT"


def test_dataset_storage_info_to_file_from_file_roundtrip(tmp_path: Path) -> None:
    storage_info = DatasetStorageInfo(
        metadata=DatasetMetadata(homepage="https://example.com"),
        split_info=SplitInfo.from_shard_info_list([make_shard()]),
    )
    path = tmp_path / "storage_info.json"
    storage_info.to_file(path)
    restored = DatasetStorageInfo.from_file(path)
    assert restored == storage_info
