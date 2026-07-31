from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from atria_core.types._utilities._url_fetchers import (
    GenericResourceQuery,
    LocalResourceLoader,
    RemoteResourceLoader,
    ResourceLoader,
    TarMemberResourceQuery,
)


def test_for_uri_local_path_dispatches_to_local_loader(tmp_path: Path) -> None:
    file_path = tmp_path / "f.txt"
    loader = ResourceLoader.for_uri(str(file_path))
    assert isinstance(loader, LocalResourceLoader)


@pytest.mark.parametrize("scheme", ["http", "https", "ftp"])
def test_for_uri_remote_schemes_dispatch_to_remote_loader(scheme: str) -> None:
    loader = ResourceLoader.for_uri(f"{scheme}://example.com/f.txt")
    assert isinstance(loader, RemoteResourceLoader)


def test_tar_member_query_can_handle() -> None:
    assert TarMemberResourceQuery.can_handle("archive.tar") is True
    assert TarMemberResourceQuery.can_handle("file.txt") is False


def test_generic_query_handles_everything() -> None:
    assert GenericResourceQuery.can_handle("anything.pdf") is True


def test_tar_member_query_byte_range_from_query_params() -> None:
    query = TarMemberResourceQuery()
    byte_range = query.byte_range({"offset": ["10"], "length": ["20"]}, "x.tar")
    assert byte_range == (10, 20)


def test_tar_member_query_missing_params_raises() -> None:
    query = TarMemberResourceQuery()
    with pytest.raises(ValueError, match="Missing or invalid"):
        query.byte_range({}, "x.tar")


def test_local_loader_reads_whole_file(tmp_path: Path) -> None:
    file_path = tmp_path / "f.txt"
    file_path.write_bytes(b"hello world")
    loader = LocalResourceLoader(str(file_path))
    assert loader.load_bytes() == b"hello world"


def test_local_loader_missing_file_raises(tmp_path: Path) -> None:
    loader = LocalResourceLoader(str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        loader.load_bytes()


def test_local_loader_tar_byte_range(tmp_path: Path) -> None:
    file_path = tmp_path / "archive.tar"
    file_path.write_bytes(b"0123456789")
    loader = LocalResourceLoader(f"{file_path}?offset=2&length=4")
    assert loader.load_bytes() == b"2345"


def test_remote_loader_whole_resource() -> None:
    loader = RemoteResourceLoader("https://example.com/f.txt")
    mock_response = MagicMock(content=b"remote bytes")
    mock_response.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_response) as mock_get:
        assert loader.load_bytes() == b"remote bytes"
        mock_get.assert_called_once_with("https://example.com/f.txt")


def test_remote_loader_byte_range_uses_range_header() -> None:
    loader = RemoteResourceLoader("https://example.com/archive.tar?offset=5&length=10")
    mock_response = MagicMock(content=b"partial bytes")
    mock_response.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_response) as mock_get:
        assert loader.load_bytes() == b"partial bytes"
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Range"] == "bytes=5-14"
