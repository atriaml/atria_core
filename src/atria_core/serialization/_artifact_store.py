from __future__ import annotations

from pathlib import Path

from atria_core.types import Image, ResourceLoader


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

    def materialize_pdf(self, key: str, source_path: str) -> str:
        """Copies the PDF at source_path into this store, returns the new path."""
        path = self.artifacts_dir / f"{key}.pdf"
        path.write_bytes(ResourceLoader.for_uri(source_path).load_bytes())
        return str(path)
