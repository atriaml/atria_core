from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np

from atria_core.types._base_data_model import BaseDataModel

_ROOT_PARENT = -1


class OCRLevel(int, enum.Enum):
    """Matches pytesseract's `level` column (1..5) directly -- no lookup table
    needed when reading tesseract's output."""

    page = 1
    block = 2
    paragraph = 3
    line = 4
    word = 5


@dataclass(frozen=True, repr=False, eq=False)
class ElementArray(BaseDataModel):
    """Every element of a page's OCR/text-layer hierarchy (page, block,
    paragraph, line, word), flat, one row per element, with `parent_ids`
    encoding the tree -- no separate segment/line-box field to store or let
    drift out of sync; a word's line box is `parent_bbox()`.

    `bboxes` is always normalized to [0, 1]. A given array field is present
    for every element or None for the whole array -- no per-element holes.
    """

    ids: np.ndarray | None = None
    parent_ids: np.ndarray | None = None
    levels: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    texts: np.ndarray | None = None  # dtype=object, elements are str
    confs: np.ndarray | None = None
    angles: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.texts is not None and not isinstance(self.texts, np.ndarray):
            raise TypeError(
                f"texts must be a numpy array (dtype=object); got {type(self.texts).__name__}"
            )

        n = len(self.texts) if self.texts is not None else None
        for name, arr in (
            ("ids", self.ids),
            ("parent_ids", self.parent_ids),
            ("levels", self.levels),
            ("bboxes", self.bboxes),
            ("confs", self.confs),
            ("angles", self.angles),
        ):
            if arr is None:
                continue
            if n is not None and len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n} (len(texts))")

        if self.bboxes is not None and self.bboxes.size:
            if self.bboxes.max() > 1.0 or self.bboxes.min() < 0.0:
                raise ValueError(
                    "bboxes must be normalized to [0, 1]: "
                    f"min={self.bboxes.min():.4f} max={self.bboxes.max():.4f}"
                )

    def validate_hierarchy(self) -> None:
        """Checks every non-root parent_id references a real id in this same
        array. Not run automatically in __post_init__: `at()`'s slices
        legitimately have parent_ids pointing outside the slice, so this is
        opt-in -- call it after building a full hierarchy (extractors do)."""
        if self.ids is None or self.parent_ids is None:
            return
        has_parent = self.parent_ids != _ROOT_PARENT
        ok = ~has_parent | np.isin(self.parent_ids, self.ids)
        if not ok.all():
            raise ValueError(
                f"dangling parent_ids at rows {np.flatnonzero(~ok)[:5].tolist()}"
            )

    @classmethod
    def from_words(cls, texts: list[str], bboxes: Any) -> ElementArray:
        """The simple, flat, no-hierarchy path: every element is a word with
        no parent. Use this when you already have OCR-shaped (text, bbox)
        pairs and don't need block/paragraph/line structure."""
        n = len(texts)
        bboxes_arr = np.asarray(bboxes, dtype=np.float64).reshape(n, 4)
        return cls(
            ids=np.arange(n),
            parent_ids=np.full(n, _ROOT_PARENT),
            levels=np.full(n, OCRLevel.word.value),
            bboxes=bboxes_arr,
            texts=np.asarray(texts, dtype=object),
        )

    def __len__(self) -> int:
        return len(self.texts) if self.texts is not None else 0

    def at(self, level: OCRLevel) -> ElementArray:
        """Only the elements at `level` (their own ids/bboxes/text are kept;
        `parent_ids` may point outside this slice -- use `segment_bboxes()`
        on the full array, not a sliced one, for that reason)."""
        if self.levels is None:
            return self
        mask = self.levels == level.value
        return ElementArray(
            ids=self.ids[mask] if self.ids is not None else None,
            parent_ids=self.parent_ids[mask] if self.parent_ids is not None else None,
            levels=self.levels[mask],
            bboxes=self.bboxes[mask] if self.bboxes is not None else None,
            texts=self.texts[mask] if self.texts is not None else None,
            confs=self.confs[mask] if self.confs is not None else None,
            angles=self.angles[mask] if self.angles is not None else None,
        )

    def parent_bbox(self) -> np.ndarray:
        """(N, 4) box of each element's parent; roots (and any element whose
        parent isn't present, e.g. after `at()`) fall back to their own box."""
        if self.bboxes is None or self.ids is None or self.parent_ids is None:
            raise ValueError("ids/parent_ids/bboxes are required to gather parent boxes")
        if len(self) == 0:
            return np.empty((0, 4))

        order = np.argsort(self.ids)
        sorted_ids = self.ids[order]
        pos = np.clip(np.searchsorted(sorted_ids, self.parent_ids), 0, len(order) - 1)
        rows = order[pos]
        out = self.bboxes[rows].copy()
        missing = (self.parent_ids == _ROOT_PARENT) | (sorted_ids[pos] != self.parent_ids)
        out[missing] = self.bboxes[missing]
        return out

    def segment_bboxes(self, level: OCRLevel = OCRLevel.line) -> np.ndarray:
        """(M, 4) enclosing box of each `level` element's parent -- the
        replacement for a stored `segment_bbox` field. Call on the full
        (unsliced) array; this gathers first, then filters to `level`."""
        if self.levels is None:
            return np.empty((0, 4))
        mask = self.levels == level.value
        return np.asarray(self.parent_bbox()[mask])

    def joined_text(self, level: OCRLevel = OCRLevel.word) -> str:
        texts = self.at(level).texts
        if texts is None:
            return ""
        return " ".join(t for t in texts.tolist() if t)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": self.ids.tolist() if self.ids is not None else None,
            "parent_ids": self.parent_ids.tolist() if self.parent_ids is not None else None,
            "levels": self.levels.tolist() if self.levels is not None else None,
            "bboxes": self.bboxes.tolist() if self.bboxes is not None else None,
            "texts": self.texts.tolist() if self.texts is not None else None,
            "confs": self.confs.tolist() if self.confs is not None else None,
            "angles": self.angles.tolist() if self.angles is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElementArray:
        def _int_array(key: str) -> np.ndarray | None:
            value = data.get(key)
            return np.asarray(value, dtype=np.int64) if value is not None else None

        def _float_array(key: str) -> np.ndarray | None:
            value = data.get(key)
            return np.asarray(value, dtype=np.float64) if value is not None else None

        texts = data.get("texts")
        return cls(
            ids=_int_array("ids"),
            parent_ids=_int_array("parent_ids"),
            levels=_int_array("levels"),
            bboxes=_float_array("bboxes"),
            texts=np.asarray(texts, dtype=object) if texts is not None else None,
            confs=_float_array("confs"),
            angles=_float_array("angles"),
        )
