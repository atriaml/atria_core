from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import tqdm

from atria_core.datasets._constants import (
    _ACCESS_TOKEN_PLACEHOLDER,
    _DOWNLOAD_TIMEOUT_SECONDS,
    _GOOGLE_DRIVE_URL_PREFIX,
    _INCOMPLETE_SUFFIX,
)
from atria_core.datasets._download._download_file_info import DownloadFileInfo
from atria_core.datasets._download._file_downloader import FileDownloader
from atria_core.logger import get_logger
from atria_core.types._utilities._repr import RepresentationMixin

logger = get_logger(__name__)


@dataclass
class UrlSpec:
    """One file to download.

    Attributes:
        url: Where to fetch from. May contain an `{access_token}` placeholder.
        url_ext: File extension to treat the download as, when it cannot be
            read off the URL path.
        rel_output_file_path: Where the file lands, relative to the data
            directory. Defaults to the file name in the URL path.
    """

    url: str
    url_ext: str | None = None
    rel_output_file_path: str | None = None


class AtriaDownloadManager(RepresentationMixin):
    """Downloads a dataset's source files and extracts any archives among them."""

    def __init__(self, data_dir: Path, download_dir: Path) -> None:
        self.data_dir = data_dir
        self.download_dir = download_dir

    def _info_from_url_spec(
        self, url_spec: UrlSpec, access_token: str | None
    ) -> DownloadFileInfo:
        """Resolve one UrlSpec into the paths its download will use.

        Raises:
            ValueError: If the output name cannot be read off a Google Drive
                URL and none was given.
        """
        rel_output_file_path = url_spec.rel_output_file_path
        if rel_output_file_path is None:
            if url_spec.url.startswith(_GOOGLE_DRIVE_URL_PREFIX):
                raise ValueError(
                    "Google Drive URLs carry no file name. Set "
                    "rel_output_file_path on the UrlSpec."
                )
            rel_output_file_path = Path(urlparse(url_spec.url).path).name

        return DownloadFileInfo(
            url=self._apply_access_token(url=url_spec.url, access_token=access_token),
            rel_output_file_path=rel_output_file_path,
            data_dir=self.data_dir,
            download_dir=self.download_dir,
            url_ext=url_spec.url_ext,
        )

    @staticmethod
    def _apply_access_token(url: str, access_token: str | None) -> str:
        if access_token is None or _ACCESS_TOKEN_PLACEHOLDER not in url:
            return url
        return url.format(access_token=access_token)

    def _download_files(self, download_file_infos: list[DownloadFileInfo]) -> None:
        for download_file_info in download_file_infos:
            if (
                download_file_info.download_path.exists()
                or download_file_info.extracted_path.exists()
                or download_file_info.output_path.exists()
            ):
                continue

            file_downloader = FileDownloader.from_url(
                parsed_url=download_file_info.parsed_url,
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            )
            file_downloader.download(download_file_info=download_file_info)

    def _merge_part_files(self, download_file_infos: list[DownloadFileInfo]) -> None:
        merged_files: dict[Path, list[str]] = {}
        for download_file_info in download_file_infos:
            if download_file_info.is_part_file:
                merged_files.setdefault(download_file_info.extractable_path, []).append(
                    str(download_file_info.download_path)
                )

        for merged_file, parts in merged_files.items():
            parts = sorted(parts, key=lambda path: int(path.split(".")[-1]))
            expected_size = sum(Path(part).stat().st_size for part in parts)

            if merged_file.exists():
                actual_size = merged_file.stat().st_size
                if actual_size == expected_size:
                    logger.info(
                        f"Skipping merge, already exists and size matches: {merged_file}"
                    )
                    continue
                else:
                    logger.warning(
                        f"Merged file {merged_file} exists but size mismatch "
                        f"(expected: {expected_size}, actual: {actual_size}), re-merging."
                    )
                    merged_file.unlink(missing_ok=True)

            logger.info(f"Merging {len(parts)} parts into {merged_file}")
            try:
                with open(merged_file, "wb") as f:
                    for part in tqdm.tqdm(parts, "Merging parts", unit="part"):
                        assert Path(part).exists(), f"Part file {part} not found"
                        with open(part, "rb") as part_file:
                            f.write(part_file.read())
            except KeyboardInterrupt:
                logger.warning("Merge interrupted by user, cleaning up.")
                merged_file.unlink(missing_ok=True)
                raise
            except Exception as e:
                merged_file.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to merge part files into {merged_file}"
                ) from e

    def _extract_archives(self, download_file_infos: list[DownloadFileInfo]) -> None:
        for download_file_info in download_file_infos:
            if (
                not download_file_info.is_compressed
                or download_file_info.extracted_path.exists()
                or download_file_info.output_path.exists()
            ):
                continue
            logger.info(
                f"Extracting {download_file_info.extractable_path} to {download_file_info.extracted_path}"
            )
            # Computed before the try block so both handlers can always clean
            # it up; an assignment inside would leave it unbound if it raised.
            incomplete_extracted_path = download_file_info.extracted_path.with_suffix(
                download_file_info.download_path.suffix + _INCOMPLETE_SUFFIX
            )
            try:
                if incomplete_extracted_path.exists():
                    incomplete_extracted_path.unlink()
                shutil.unpack_archive(
                    filename=download_file_info.extractable_path,
                    extract_dir=incomplete_extracted_path,
                )
                incomplete_extracted_path.rename(download_file_info.extracted_path)
            except KeyboardInterrupt:
                logger.warning("Extraction interrupted by user, cleaning up.")
                if incomplete_extracted_path.exists():
                    shutil.rmtree(incomplete_extracted_path)
                raise
            except Exception as e:
                if incomplete_extracted_path.exists():
                    shutil.rmtree(incomplete_extracted_path)
                raise RuntimeError(
                    f"Failed to extract {download_file_info.extractable_path}"
                ) from e

    def _finalize_files(
        self, download_file_infos: list[DownloadFileInfo], extract: bool = True
    ) -> None:
        for download_file_info in download_file_infos:
            logger.info("Finalizing file: %s", download_file_info.download_path)
            if download_file_info.output_path.exists():
                continue
            if download_file_info.is_part_file:
                continue
            if download_file_info.is_compressed and extract:
                if download_file_info.extracted_path.exists():
                    logger.debug(
                        f"Removing compressed file {download_file_info.download_path}"
                    )
                    download_file_info.extractable_path.unlink(missing_ok=True)
                shutil.move(
                    download_file_info.extracted_path, download_file_info.output_path
                )
            else:
                logger.debug(
                    f"Moving {download_file_info.download_path} to {download_file_info.output_path}"
                )
                shutil.move(
                    download_file_info.download_path, download_file_info.output_path
                )

    def download_and_extract(
        self,
        data_urls: list[UrlSpec],
        extract: bool = True,
        access_token: str | None = None,
    ) -> dict[str, Path]:
        """Download every file, optionally extracting archives among them.

        Args:
            data_urls: Files to fetch.
            extract: Whether to unpack downloaded archives.
            access_token: Substituted into URLs containing `{access_token}`.

        Returns:
            Each downloaded file's final path, keyed by its name.
        """
        download_file_infos = [
            self._info_from_url_spec(url_spec=url_spec, access_token=access_token)
            for url_spec in data_urls
        ]
        if extract:
            for download_file_info in download_file_infos:
                download_file_info.update_extract_path()

        if any(
            not download_file_info.is_download_completed
            for download_file_info in download_file_infos
        ):
            logger.info(
                f"Downloading {len(download_file_infos)} files to {self.download_dir}"
            )
            self._download_files(download_file_infos=download_file_infos)
            self._merge_part_files(download_file_infos)
            if extract:
                self._extract_archives(download_file_infos)
            self._finalize_files(download_file_infos, extract=extract)
        return {
            download_file_info.output_path.name: download_file_info.output_path
            for download_file_info in download_file_infos
        }
