from __future__ import annotations

import functools
import threading
from collections.abc import Iterator
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from atria_core.transforms import functional as F
from atria_core.types import (
    Image,
    MultiPageDocumentInstance,
    RemoteResourceLoader,
    ResourceLoader,
)

# These tests exercise the RemoteResourceLoader/requests code path against a
# real HTTP server, rather than a third-party URL: that keeps the "fetch over
# the network" integration test deterministic and CI-safe without depending
# on any live external host or a URL we'd otherwise have to hardcode/guess.


@pytest.fixture
def http_server(tmp_path: Path) -> Iterator[str]:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_remote_resource_loader_fetches_bytes(tmp_path: Path, http_server: str) -> None:
    (tmp_path / "data.bin").write_bytes(b"hello over http")

    loader = ResourceLoader.for_uri(f"{http_server}/data.bin")
    assert isinstance(loader, RemoteResourceLoader)
    assert loader.load_bytes() == b"hello over http"


def test_fetch_process_use_remote_image(tmp_path: Path, http_server: str) -> None:
    """Use case C: fetch an image from an online source, process it, use it."""
    original = PILImage.new("RGB", (20, 10), color="red")
    original.save(tmp_path / "photo.png")

    image = Image.from_source(f"{http_server}/photo.png")
    assert image.content is None  # not fetched yet

    image = image.load()
    assert np.array_equal(np.array(original), np.array(image.require_content()))

    processed = F.image.resize(image, 10, 5)
    assert processed.size == (10, 5)


def test_fetch_remote_multi_page_pdf(
    tmp_path: Path, http_server: str, sample_pdf_path: Path
) -> None:
    """Use case C, PDF variant: a MultiPageDocumentInstance sourced from a URL
    rather than a local path -- bytes are fetched fresh each time a page
    renders."""
    remote_pdf = tmp_path / "remote.pdf"
    remote_pdf.write_bytes(sample_pdf_path.read_bytes())

    document = MultiPageDocumentInstance(
        sample_id="remote", source_path=f"{http_server}/remote.pdf"
    )
    assert document.num_pages == 2

    page = document.get_page(0)
    assert page.page_id == 0
    assert page.visual.load().require_content().size[0] > 0
