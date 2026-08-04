from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from atria_core.types._base_data_model import BaseDataModel
from atria_core.types._generic._bounding_box import BoundingBoxMode

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

    `bbox_mode` and `normalized` describe every box in the array. A given
    array field is present for every element or None for the whole array --
    no per-element holes.
    """

    ids: np.ndarray | None = None
    parent_ids: np.ndarray | None = None
    levels: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    bbox_mode: BoundingBoxMode = BoundingBoxMode.XYXY
    normalized: bool = False
    texts: np.ndarray | None = None  # dtype=object, elements are str
    confs: np.ndarray | None = None
    angles: np.ndarray | None = None
    segmentations: np.ndarray | None = None  # (N, P_max, 2), NaN-padded
    segmentation_lengths: np.ndarray | None = None  # (N,) real point count per element

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
            ("segmentations", self.segmentations),
            ("segmentation_lengths", self.segmentation_lengths),
        ):
            if arr is None:
                continue
            if n is not None and len(arr) != n:
                raise ValueError(
                    f"{name} has length {len(arr)}, expected {n} (len(texts))"
                )

        if self.normalized and self.bboxes is not None and self.bboxes.size:
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
    def from_words(
        cls,
        texts: list[str],
        bboxes: Any,
        segmentations: list[np.ndarray | None] | None = None,
        *,
        bbox_mode: BoundingBoxMode = BoundingBoxMode.XYXY,
        normalized: bool = False,
    ) -> ElementArray:
        """The simple, flat, no-hierarchy path: every element is a word with
        no parent. Use this when you already have OCR-shaped (text, bbox)
        pairs and don't need block/paragraph/line structure. `segmentations`,
        if given, is one (P, 2) polygon (or None) per word -- ragged point
        counts are NaN-padded into a single (N, P_max, 2) array."""
        n = len(texts)
        bboxes_arr = np.asarray(bboxes, dtype=np.float64).reshape(n, 4)

        segmentations_arr = None
        segmentation_lengths_arr = None
        if segmentations is not None:
            lengths = np.array([0 if p is None else len(p) for p in segmentations])
            p_max = int(lengths.max()) if len(lengths) else 0
            padded = np.full((n, p_max, 2), np.nan)
            for i, polygon in enumerate(segmentations):
                if polygon is not None:
                    padded[i, : len(polygon)] = polygon
            segmentations_arr = padded
            segmentation_lengths_arr = lengths

        return cls(
            ids=np.arange(n),
            parent_ids=np.full(n, _ROOT_PARENT),
            levels=np.full(n, OCRLevel.word.value),
            bboxes=bboxes_arr,
            bbox_mode=bbox_mode,
            normalized=normalized,
            texts=np.asarray(texts, dtype=object),
            segmentations=segmentations_arr,
            segmentation_lengths=segmentation_lengths_arr,
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
        return replace(
            self,
            ids=self.ids[mask] if self.ids is not None else None,
            parent_ids=self.parent_ids[mask] if self.parent_ids is not None else None,
            levels=self.levels[mask],
            bboxes=self.bboxes[mask] if self.bboxes is not None else None,
            texts=self.texts[mask] if self.texts is not None else None,
            confs=self.confs[mask] if self.confs is not None else None,
            angles=self.angles[mask] if self.angles is not None else None,
            segmentations=self.segmentations[mask]
            if self.segmentations is not None
            else None,
            segmentation_lengths=self.segmentation_lengths[mask]
            if self.segmentation_lengths is not None
            else None,
        )

    def box_batches(self) -> dict[str, np.ndarray | None]:
        return {"bboxes": self.bboxes}

    def with_box_batches(
        self,
        batches: dict[str, np.ndarray],
        *,
        normalized: bool,
        mode: BoundingBoxMode,
    ) -> ElementArray:
        return replace(
            self,
            bboxes=batches.get("bboxes", self.bboxes),
            normalized=normalized,
            bbox_mode=mode,
        )

    def parent_bbox(self) -> np.ndarray:
        """(N, 4) box of each element's parent; roots (and any element whose
        parent isn't present, e.g. after `at()`) fall back to their own box."""
        if self.bboxes is None or self.ids is None or self.parent_ids is None:
            raise ValueError(
                "ids/parent_ids/bboxes are required to gather parent boxes"
            )
        if len(self) == 0:
            return np.empty((0, 4))

        order = np.argsort(self.ids)
        sorted_ids = self.ids[order]
        pos = np.clip(np.searchsorted(sorted_ids, self.parent_ids), 0, len(order) - 1)
        rows = order[pos]
        out = self.bboxes[rows].copy()
        missing = (self.parent_ids == _ROOT_PARENT) | (
            sorted_ids[pos] != self.parent_ids
        )
        out[missing] = self.bboxes[missing]
        return out

    def word_bboxes(self) -> np.ndarray:
        """(M, 4) boxes of every word in this array, in order. If the array has
        no words, returns an empty (0, 4) array."""
        if self.levels is None or self.bboxes is None:
            return np.empty((0, 4))
        mask = self.levels == OCRLevel.word.value
        return np.asarray(self.bboxes[mask])

    def word_texts(self) -> list[str]:
        """List of every word's text in this array, in order. If the array has
        no words, returns an empty list."""
        if self.levels is None or self.texts is None:
            return []
        mask = self.levels == OCRLevel.word.value
        return [t for t in self.texts[mask].tolist() if t]

    def segment_bboxes(self, level: OCRLevel = OCRLevel.line) -> np.ndarray:
        """(M, 4) enclosing box of each `level` element's parent -- the
        replacement for a stored `segment_bbox` field. Call on the full
        (unsliced) array; this gathers first, then filters to `level`."""
        if self.levels is None:
            return np.empty((0, 4))
        mask = self.levels == level.value
        return np.asarray(self.parent_bbox()[mask])

    def joined_text(self, level: OCRLevel = OCRLevel.word, sep: str = " ") -> str:
        texts = self.at(level).texts
        if texts is None:
            return ""
        return sep.join(t for t in texts.tolist() if t)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": self.ids.tolist() if self.ids is not None else None,
            "parent_ids": self.parent_ids.tolist()
            if self.parent_ids is not None
            else None,
            "levels": self.levels.tolist() if self.levels is not None else None,
            "bboxes": self.bboxes.tolist() if self.bboxes is not None else None,
            "bbox_mode": self.bbox_mode.value,
            "normalized": self.normalized,
            "texts": self.texts.tolist() if self.texts is not None else None,
            "confs": self.confs.tolist() if self.confs is not None else None,
            "angles": self.angles.tolist() if self.angles is not None else None,
            "segmentations": self.segmentations.tolist()
            if self.segmentations is not None
            else None,
            "segmentation_lengths": self.segmentation_lengths.tolist()
            if self.segmentation_lengths is not None
            else None,
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
            bbox_mode=BoundingBoxMode(
                data.get("bbox_mode", BoundingBoxMode.XYXY.value)
            ),
            # ElementArray snapshots written before this metadata existed
            # always used normalized coordinates.
            normalized=data.get("normalized", True),
            texts=np.asarray(texts, dtype=object) if texts is not None else None,
            confs=_float_array("confs"),
            angles=_float_array("angles"),
            segmentations=_float_array("segmentations"),
            segmentation_lengths=_int_array("segmentation_lengths"),
        )
