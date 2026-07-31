import ast
from pathlib import Path
from typing import Annotated, Any

from pydantic import field_serializer, field_validator

from atria_core.types._base._data_model import BaseDataModel
from atria_core.types._common import OCRType
from atria_core.types._pydantic import OptStrField, TableSchemaMetadata
from atria_core.types._utilities._url_fetchers import _load_bytes_from_uri


def _detect_encoding(content: bytes) -> str:
    # Compressed formats
    if content[:2] == b"\x1f\x8b":
        return "gzip"
    if content[:2] == b"x\x9c" or content[:2] == b"x\x01" or content[:2] == b"x\xda":
        return "zlib"
    if content[:4] == b"\x28\xb5\x2f\xfd":
        return "zstd"
    if content[:3] == b"BZh":
        return "bzip2"

    # Plain text
    return "utf-8"


class OCR(BaseDataModel):
    file_path: OptStrField = None
    type: Annotated[OCRType | None, TableSchemaMetadata(pa_type="string")] = None
    content: Annotated[str | None, TableSchemaMetadata(pa_type="binary")] = None

    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, value: Any) -> str | None:
        if isinstance(value, Path):
            return str(value)
        return value

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, value: Any) -> str | None:
        if isinstance(value, str):
            return OCRType(value)
        return value

    def load(self):
        from atria_core.types._utilities._string_encoding import _decompress_string

        if self.content is None:
            if self.file_path is None:
                raise ValueError("Either file_path or content must be provided.")

            content = _load_bytes_from_uri(self.file_path)
            if content.startswith(b"b'"):
                content = ast.literal_eval(content.decode("utf-8"))
            if isinstance(content, bytes):
                encoding = _detect_encoding(content)
                if encoding == "utf-8":
                    content = content.decode("utf-8")
                else:
                    content = _decompress_string(
                        content
                    )  # already knows it's compressed

            return self.model_copy(
                update={
                    "content": content,
                }
            )
        return self

    @field_serializer("content")
    def _serialize_content(self, value: str | None) -> bytes | None:
        from atria_core.types._utilities._string_encoding import _compress_string

        if value is None:
            return None
        return _compress_string(value)

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: Any) -> str | None:
        from atria_core.types._utilities._string_encoding import _decompress_string

        if value is None:
            return None
        if isinstance(value, bytes):
            return _decompress_string(value)
        return value
