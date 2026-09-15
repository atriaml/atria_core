from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import ParseResult, urlparse, urlunparse

from atria_core.types._utilities._repr import RepresentationMixin

_SUPPORTED_URLS = ["http", "https", "ftp"]
_COMPRESSED_FILES_REGEX = r"\.(zip|tar|tar\.gz|tgz)(\..+)?$"


class DownloadFileInfo(RepresentationMixin):
    """Every path derived from one download URL: where it downloads, extracts,
    and finally lands."""

    def __init__(
        self,
        url: str,
        rel_output_file_path: str,
        data_dir: Path,
        download_dir: Path,
        url_ext: str | None = None,
    ) -> None:
        self.url = url
        self.rel_output_file_path = rel_output_file_path
        self.data_dir = Path(data_dir)
        self.download_dir = Path(download_dir)
        self.url_ext = url_ext
        parsed = urlparse(self.url)
        if parsed.scheme == "":
            raise ValueError(
                f"URL {url} is invalid. URL must have a scheme (http, https, ftp)."
            )
        if parsed.scheme not in _SUPPORTED_URLS:
            raise ValueError(
                f"URL {url} is not supported. Supported URL schemes are: {', '.join(_SUPPORTED_URLS)}"
            )
        download_dir.mkdir(parents=True, exist_ok=True)

    def update_extract_path(self) -> None:
        """Strip the archive suffix from the output path, for compressed downloads."""
        if self.is_compressed:
            match = re.search(_COMPRESSED_FILES_REGEX, self.rel_output_file_path)
            if match:
                self.rel_output_file_path = self.rel_output_file_path.replace(
                    match.group(), ""
                )

    @property
    def parsed_url(self) -> ParseResult:
        """The download URL, parsed into components."""
        return urlparse(self.url)

    @property
    def hashed_url(self) -> str:
        """Stable digest of the full URL, used as its on-disk download name."""
        return hashlib.sha256(self.url.encode()).hexdigest()

    @property
    def hashed_url_without_part(self) -> str:
        """Digest of the URL minus its part suffix, shared by all parts of one archive."""
        url_without_part = urlunparse((
            self.parsed_url.scheme,
            self.parsed_url.netloc,
            str(Path(self.parsed_url.path).with_suffix("")),
            self.parsed_url.params,
            self.parsed_url.query,
            self.parsed_url.fragment,
        ))
        return hashlib.sha256(url_without_part.encode()).hexdigest()

    @property
    def url_path_ext(self) -> str:
        """The URL's file extension, or the override given at construction."""
        return (
            "".join(Path(self.parsed_url.path).suffixes)
            if self.url_ext is None
            else self.url_ext
        )

    @property
    def download_path(self) -> Path:
        """Where the raw bytes for this URL are downloaded to."""
        return self.download_dir / (self.hashed_url + self.url_path_ext)

    @property
    def extractable_path(self) -> Path:
        """For part files (e.g. `.zip.001`) this is the merged-file path, not the part's own path."""
        if self.is_part_file:
            return self.download_dir / (
                self.hashed_url_without_part + Path(self.parsed_url.path).suffixes[-2]
            )
        else:
            return self.download_path

    @property
    def extracted_path(self) -> Path:
        """Directory this download's archive is extracted into."""
        if self.is_part_file:
            return self.data_dir / (self.hashed_url_without_part)
        else:
            return self.data_dir / self.hashed_url

    @property
    def is_part_file(self) -> bool:
        """Whether this URL is one numbered part of a split archive."""
        return bool(re.search(r"\.(zip|tar|tar\.gz|tgz)\.\d+$", self.parsed_url.path))

    @property
    def is_compressed(self) -> bool:
        """Whether this download is an archive needing extraction."""
        if self.url_ext is not None:
            return bool(
                re.search(_COMPRESSED_FILES_REGEX, self.parsed_url.path)
            ) or bool(re.search(_COMPRESSED_FILES_REGEX, self.url_ext))
        else:
            return bool(re.search(_COMPRESSED_FILES_REGEX, self.parsed_url.path))

    @property
    def output_path(self) -> Path:
        """Final location this download lands at, under the data directory."""
        return self.data_dir / self.rel_output_file_path

    @property
    def is_download_completed(self) -> bool:
        """Whether the final output for this URL already exists."""
        return self.data_dir.exists() and self.output_path.exists()
