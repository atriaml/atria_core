from __future__ import annotations

from pathlib import Path

from PIL.Image import Image as PILImage

from atria_core.types._generic._image import Image
from atria_core.types._utilities._url_fetchers import ResourceLoader


class ArtifactStore:
    """Writes/reads binary artifacts (image/PDF bytes) as separate files under
    <root>/artifacts/, keyed by sample key. Keeps parquet rows free of large
    binary blobs -- rows only ever hold relative path references. Always
    copies bytes in, regardless of where they started, so the dataset
    directory is self-contained and portable."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def materialize_image(self, key: str, image: Image) -> Image:
        """Writes `image`'s content into this store and returns a new,
        unloaded, file-backed Image pointing at the copy."""
        image = image.load()
        path = self.artifacts_dir / f"{key}.png"
        image.require_content().save(path)
        return Image.from_source(str(path))

    def materialize_page_image(self, key: str, image: PILImage) -> str:
        """Same as materialize_image, but for a raw PIL image (as held by
        SinglePageDocument, which doesn't wrap it in an `Image`)."""
        path = self.artifacts_dir / f"{key}.png"
        image.save(path)
        return str(path)

    def materialize_pdf(self, key: str, source_path: str) -> str:
        """Copies the PDF at source_path into this store, returns the new path."""
        path = self.artifacts_dir / f"{key}.pdf"
        path.write_bytes(ResourceLoader.for_uri(source_path).load_bytes())
        return str(path)
