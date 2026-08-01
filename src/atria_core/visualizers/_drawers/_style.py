from __future__ import annotations

from dataclasses import dataclass, field

#: 0-255 RGB tuples (PIL-native). PdfDrawer divides by 255 for pymupdf, which
#: wants 0-1 floats.
_DEFAULT_COLOR_PALETTE = (
    (255, 99, 71),  # tomato
    (255, 140, 0),  # dark orange
    (255, 165, 0),  # orange
    (255, 69, 0),  # red-orange
    (255, 215, 0),  # gold
    (255, 182, 80),  # light orange
)


@dataclass(frozen=True, repr=False)
class DrawStyle:
    """Everything about *how* a bbox+label is drawn -- shared by every
    BboxDrawer implementation so appearance stays consistent regardless of
    target (image vs. PDF page)."""

    colors: tuple[tuple[int, int, int], ...] = field(default=_DEFAULT_COLOR_PALETTE)
    bbox_width: int = 2
    text_color: tuple[int, int, int] = (255, 255, 255)
    text_background: tuple[int, int, int, int] = (0, 0, 0, 180)
    text_padding: int = 3
    label_offset: int = 4
    min_font_size: int = 12
    font_size_ratio: float = 0.015

    def font_size(self, canvas_height: float) -> int:
        return max(self.min_font_size, int(canvas_height * self.font_size_ratio))


DEFAULT_STYLE = DrawStyle()
