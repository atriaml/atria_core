from __future__ import annotations

import random

import numpy as np
import pymupdf

from atria_core.visualizers._drawers._style import DEFAULT_STYLE, DrawStyle

_FONT_NAME = "helv"


def _to_unit_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255, color[1] / 255, color[2] / 255)


def _create_label_color_mapping(
    labels: list[str] | None, style: DrawStyle
) -> dict[str, tuple[int, int, int]]:
    if not labels:
        return {}
    unique_labels = set(labels)
    return {
        label: style.colors[idx % len(style.colors)]
        for idx, label in enumerate(unique_labels)
    }


def _get_bbox_color(
    label: str | None,
    label_to_color: dict[str, tuple[int, int, int]],
    style: DrawStyle,
) -> tuple[int, int, int]:
    if label and label in label_to_color:
        return label_to_color[label]
    return random.choice(style.colors)


def _draw_text_label(
    page: pymupdf.Page,
    text: str,
    rect: pymupdf.Rect,
    font_size: float,
    style: DrawStyle,
) -> None:
    text_width = pymupdf.get_text_length(text, fontsize=font_size, fontname=_FONT_NAME)
    text_height = font_size

    label_y1 = max(0, rect.y0 - style.label_offset)
    label_y0 = max(0, label_y1 - text_height - style.label_offset)
    background_rect = pymupdf.Rect(  # type: ignore[no-untyped-call]
        rect.x0,
        label_y0,
        rect.x0 + text_width + (2 * style.text_padding),
        label_y1,
    )
    page.draw_rect(
        background_rect,
        color=None,
        fill=_to_unit_color(style.text_background[:3]),
        fill_opacity=style.text_background[3] / 255,
    )
    page.insert_text(
        pymupdf.Point(  # type: ignore[no-untyped-call]
            rect.x0 + style.text_padding, label_y1 - style.text_padding
        ),
        text,
        fontsize=font_size,
        fontname=_FONT_NAME,
        color=_to_unit_color(style.text_color),
    )


class PdfDrawer:
    """BboxDrawer for pymupdf Pages. Mutates and returns the same Page --
    pymupdf pages are handles into an open Document, not value types, same
    "no cheap immutable copy" reasoning as ImageDrawer."""

    def draw(
        self,
        target: pymupdf.Page,
        bboxes: np.ndarray,
        *,
        texts: list[str] | None = None,
        labels: list[str] | None = None,
        style: DrawStyle = DEFAULT_STYLE,
    ) -> pymupdf.Page:
        label_to_color = _create_label_color_mapping(labels, style)
        font_size = style.font_size(target.rect.height)

        for index, bbox in enumerate(bboxes):
            text = texts[index] if texts else None
            label = labels[index] if labels else None
            color = _get_bbox_color(label, label_to_color, style)

            rect = pymupdf.Rect(*bbox.tolist())  # type: ignore[no-untyped-call]
            target.draw_rect(rect, color=_to_unit_color(color), width=style.bbox_width)

            combined = None
            if label is not None:
                combined = f"{label}: {text}" if text is not None else label
            elif text is not None:
                combined = text

            if combined:
                _draw_text_label(target, combined, rect, font_size, style)

        return target
