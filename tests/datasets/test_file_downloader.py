from __future__ import annotations

from urllib.parse import urlparse

import pytest

from atria_core.datasets._download._file_downloader import (
    FileDownloader,
    FTPFileDownloader,
    GoogleDriveDownloader,
    HTTPDownloader,
)


def test_from_url_dispatches_http() -> None:
    downloader = FileDownloader.from_url(urlparse("http://example.com/file.tar.gz"))
    assert isinstance(downloader, HTTPDownloader)


def test_from_url_dispatches_https() -> None:
    downloader = FileDownloader.from_url(urlparse("https://example.com/file.tar.gz"))
    assert isinstance(downloader, HTTPDownloader)


def test_from_url_dispatches_ftp() -> None:
    downloader = FileDownloader.from_url(urlparse("ftp://example.com/file.tar.gz"))
    assert isinstance(downloader, FTPFileDownloader)


def test_from_url_dispatches_google_drive() -> None:
    downloader = FileDownloader.from_url(
        urlparse("https://drive.google.com/uc?id=abc123")
    )
    assert isinstance(downloader, GoogleDriveDownloader)


def test_from_url_rejects_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported download URL"):
        FileDownloader.from_url(urlparse("s3://bucket/file.tar.gz"))
