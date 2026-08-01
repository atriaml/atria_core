from __future__ import annotations

import random

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from atria_core.visualizers._drawers._style import DEFAULT_STYLE, DrawStyle


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


def _get_text_dimensions(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        width = text_bbox[2] - text_bbox[0]
        height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        width, height = font.getsize(text)  # type: ignore
    return width, height


def _get_bbox_color(
    label: str | None,
    label_to_color: dict[str, tuple[int, int, int]],
    style: DrawStyle,
) -> tuple[int, int, int]:
    if label and label in label_to_color:
        return label_to_color[label]
    return random.choice(style.colors)


def _draw_bbox_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: np.ndarray,
    color: tuple[int, int, int],
    style: DrawStyle,
) -> None:
    outline_color = (*color, 255)
    draw.rectangle(tuple(bbox.tolist()), outline=outline_color, width=style.bbox_width)


def _draw_text_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: np.ndarray,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    style: DrawStyle,
) -> None:
    text_width, text_height = _get_text_dimensions(draw, text, font)

    text_x = bbox[0]
    text_y = max(0, bbox[1] - text_height - style.label_offset)

    background_bbox = [
        text_x,
        text_y,
        text_x + text_width + (2 * style.text_padding),
        text_y + text_height + style.label_offset,
    ]
    draw.rectangle(background_bbox, fill=style.text_background)

    text_position = (text_x + style.text_padding, text_y + 2)
    draw.text(text_position, text, fill=(*style.text_color, 255), font=font)


class ImageDrawer:
    """BboxDrawer for PIL images. Mutates and returns the same Image --
    drawing is inherently in-place on a PIL canvas, there's no cheap
    value-copy alternative."""

    def draw(
        self,
        target: PILImage.Image,
        bboxes: np.ndarray,
        *,
        texts: list[str] | None = None,
        labels: list[str] | None = None,
        style: DrawStyle = DEFAULT_STYLE,
    ) -> PILImage.Image:
        label_to_color = _create_label_color_mapping(labels, style)
        draw = ImageDraw.Draw(target, "RGBA")
        font = ImageFont.load_default()

        for index, bbox in enumerate(bboxes):
            text = texts[index] if texts else None
            label = labels[index] if labels else None
            color = _get_bbox_color(label, label_to_color, style)

            _draw_bbox_rectangle(draw, bbox, color, style)

            if label is not None:
                combined = f"{label}: {text}" if text is not None else label
                _draw_text_label(draw, combined, bbox, font, style)
            elif text is not None:
                _draw_text_label(draw, text, bbox, font, style)

        return target
