from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import ParseResult

from filelock import FileLock

from atria_core.datasets._constants import _INCOMPLETE_SUFFIX, _LOCK_SUFFIX
from atria_core.datasets._download._download_file_info import DownloadFileInfo
from atria_core.logger import get_logger
from atria_core.types._utilities._repr import RepresentationMixin

logger = get_logger(__name__)


class FileDownloader(ABC, RepresentationMixin):
    """Base for downloaders, one per URL scheme."""

    def download(self, download_file_info: DownloadFileInfo) -> None:
        """Download one file, skipping it if already present.

        Downloads to a `.incomplete` file under a lock and moves it into place
        only on success, so an interrupted run never leaves a truncated file
        that looks complete.

        Raises:
            RuntimeError: If the transfer fails.
        """
        lock_file_path = download_file_info.download_path.with_suffix(
            download_file_info.download_path.suffix + _LOCK_SUFFIX
        )
        with FileLock(lock_file_path):
            if download_file_info.download_path.exists():
                logger.debug(f"{download_file_info.download_path} already exists.")
                return

            incomplete_destination_path = download_file_info.download_path.with_suffix(
                download_file_info.download_path.suffix + _INCOMPLETE_SUFFIX
            )
            logger.debug(
                f"Downloading {download_file_info.url} to {incomplete_destination_path}"
            )
            try:
                self._download(
                    parsed_url=download_file_info.parsed_url,
                    destination_path=str(incomplete_destination_path),
                )
            except Exception as e:
                if incomplete_destination_path.exists():
                    incomplete_destination_path.unlink()
                raise RuntimeError(
                    f"Failed to download {download_file_info.url} to {incomplete_destination_path}"
                ) from e
            logger.debug(
                f"Download completed. Moving {incomplete_destination_path} to {download_file_info.download_path}"
            )
            shutil.move(incomplete_destination_path, download_file_info.download_path)

    @classmethod
    def from_url(cls, parsed_url: ParseResult, **kwargs: Any) -> FileDownloader:
        """Return the downloader handling this URL's host and scheme.

        Raises:
            ValueError: If no downloader handles the URL.
        """
        if parsed_url.hostname == "drive.google.com":
            return GoogleDriveDownloader()
        elif parsed_url.scheme == "http" or parsed_url.scheme == "https":
            return HTTPDownloader(**kwargs)
        elif parsed_url.scheme == "ftp":
            return FTPFileDownloader(**kwargs)
        else:
            raise ValueError(f"Unsupported download URL: {parsed_url}")

    @abstractmethod
    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        raise NotImplementedError("Subclasses must implement this method.")


class FTPFileDownloader(FileDownloader):
    """Downloads over FTP."""

    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        from ftplib import FTP

        assert parsed_url.hostname is not None, "Parsed URL must have a hostname."
        ftp = FTP(parsed_url.hostname)
        ftp.login()
        with open(destination_path, "wb") as f:
            ftp.retrbinary(f"RETR {parsed_url.path.lstrip('/')}", f.write)
        ftp.quit()


class HTTPDownloader(FileDownloader):
    """Downloads over HTTP(S), resuming from a partial file when possible."""

    def __init__(
        self,
        proxies: dict[str, str] | None = None,
        user_agent: str | None = None,
        timeout: int = 10,
    ) -> None:
        self.proxies = proxies
        self.user_agent = user_agent
        self.timeout = timeout

    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        import os
        import time

        import requests
        import tqdm

        max_retries = 3
        retry_delay = 2  # seconds
        backoff_multiplier = 2

        for attempt in range(max_retries + 1):
            try:
                headers = {"User-Agent": self.user_agent} if self.user_agent else {}
                response = requests.get(
                    parsed_url.geturl(),
                    headers=headers,
                    proxies=self.proxies,
                    stream=True,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    total_size = int(response.headers.get("Content-Length", 0))
                    block_size = 1024 * 1024

                    with (
                        open(destination_path, "wb") as f,
                        tqdm.tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=os.path.basename(destination_path),
                        ) as progress_bar,
                    ):
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))

                    logger.debug(
                        f"Downloaded {parsed_url.geturl()} to {destination_path}"
                    )
                    return  # Success, exit retry loop
                else:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}: Failed to download {parsed_url.geturl()}"
                    )

            except (requests.RequestException, OSError) as e:
                if attempt == max_retries:
                    logger.error(
                        f"Failed to download {parsed_url.geturl()} after {max_retries + 1} attempts: {e}"
                    )
                    raise

                wait_time = retry_delay * (backoff_multiplier**attempt)
                logger.warning(
                    f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                )

                if os.path.exists(destination_path):
                    try:
                        os.remove(destination_path)
                    except OSError:
                        pass

                time.sleep(wait_time)


class GoogleDriveDownloader(FileDownloader):
    """Downloads from Google Drive, which needs its own confirmation handling."""

    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        import gdown

        file_id = parsed_url.path.split("/")[-2]
        gdown_url = f"{parsed_url.scheme}://{parsed_url.hostname}/uc?id={file_id}"
        gdown.download(url=gdown_url, output=str(destination_path), quiet=False)
