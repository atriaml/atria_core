from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import ParseResult, parse_qs, urlencode, urlparse, urlunparse

import requests

_REMOTE_SCHEMES = ("http", "https", "ftp")


class ResourceQuery(ABC):
    """Knows how to turn a parsed URI's query string into a byte range to
    fetch (or `None` for "the whole resource"), for one file type. This is
    the only thing that varies between file types -- how the resource is
    physically fetched (local vs. remote) doesn't depend on it."""

    @classmethod
    @abstractmethod
    def can_handle(cls, path: str) -> bool: ...

    @abstractmethod
    def byte_range(self, query: dict[str, list[str]], uri: str) -> tuple[int, int] | None: ...


class TarMemberResourceQuery(ResourceQuery):
    """*.tar + ?offset=&length= -> that byte range."""

    @classmethod
    def can_handle(cls, path: str) -> bool:
        return path.endswith(".tar")

    def byte_range(self, query: dict[str, list[str]], uri: str) -> tuple[int, int]:
        try:
            offset = int(query["offset"][0])
            length = int(query["length"][0])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Missing or invalid 'offset' and 'length' in tar URI: {uri}"
            ) from None
        return offset, length


class GenericResourceQuery(ResourceQuery):
    """Anything else -> fetch the whole resource."""

    @classmethod
    def can_handle(cls, path: str) -> bool:
        return True

    def byte_range(self, query: dict[str, list[str]], uri: str) -> None:
        return None


# Order matters: more specific file types are tried before the generic
# fallback. Adding a new file type later is just a new ResourceQuery
# subclass + one entry here.
_RESOURCE_QUERIES: list[type[ResourceQuery]] = [TarMemberResourceQuery, GenericResourceQuery]


def _get_resource_query(path: str) -> ResourceQuery:
    for query_cls in _RESOURCE_QUERIES:
        if query_cls.can_handle(path):
            return query_cls()
    raise ValueError(f"Unsupported file type for path: {path}")  # unreachable: Generic always matches


class ResourceLoader(ABC):
    """Fetches the raw bytes a URI points at. Subclasses only differ in
    *where* the resource lives (local disk vs. remote network); *what* to
    fetch (whole file vs. a byte range) is delegated to a `ResourceQuery`."""

    def __init__(self, uri: str):
        self.uri = uri
        self.parsed: ParseResult = urlparse(uri)
        self.path = self.parsed.path
        self.query = parse_qs(self.parsed.query)
        self._resource_query = _get_resource_query(self.path)

    @property
    def byte_range(self) -> tuple[int, int] | None:
        return self._resource_query.byte_range(self.query, self.uri)

    @abstractmethod
    def load_bytes(self) -> bytes: ...

    @classmethod
    def for_uri(cls, uri: str) -> ResourceLoader:
        is_remote = urlparse(uri).scheme in _REMOTE_SCHEMES
        loader_cls = RemoteResourceLoader if is_remote else LocalResourceLoader
        return loader_cls(uri)


class LocalResourceLoader(ResourceLoader):
    def load_bytes(self) -> bytes:
        local_path = Path(self.path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        byte_range = self.byte_range
        if byte_range is None:
            if not local_path.is_file():
                raise ValueError(f"Provided path is not a file: {local_path}")
            return local_path.read_bytes()

        offset, length = byte_range
        with open(local_path, "rb") as f:
            f.seek(offset)
            return f.read(length)


class RemoteResourceLoader(ResourceLoader):
    def load_bytes(self) -> bytes:
        byte_range = self.byte_range
        if byte_range is None:
            response = requests.get(self.uri)
            response.raise_for_status()
            return response.content

        offset, length = byte_range
        url = self._strip_query_params("offset", "length")
        response = requests.get(
            url,
            headers={"Range": f"bytes={offset}-{offset + length - 1}"},
            stream=True,
        )
        response.raise_for_status()
        return response.content

    def _strip_query_params(self, *keys: str) -> str:
        query = {k: v for k, v in self.query.items() if k not in keys}
        new_query = urlencode(query, doseq=True)
        return urlunparse(
            (
                self.parsed.scheme,
                self.parsed.netloc,
                self.parsed.path,
                self.parsed.params,
                new_query,
                self.parsed.fragment,
            )
        )
