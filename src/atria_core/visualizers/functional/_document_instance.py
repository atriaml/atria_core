from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image as PILImage

from atria_core.logger import get_logger
from atria_core.types import (
    AnnotationType,
    BoundingBoxMode,
    DocumentContent,
    DocumentInstance,
    MultiPageDocumentInstance,
    OCRLevel,
    PdfPage,
    ResourceLoader,
    SinglePageDocumentInstance,
)
from atria_core.types._arrays import FloatArray
from atria_core.types._generic._annotations import (
    ObjectDetectionAnnotation,
    OCRAnnotation,
)
from atria_core.visualizers._drawers._image import ImageDrawer
from atria_core.visualizers._drawers._pdf import PdfDrawer
from atria_core.visualizers._drawers._style import DEFAULT_STYLE, DrawStyle
from atria_core.visualizers.functional._visualize import output_name

logger = get_logger(__name__)


_DrawLayer = tuple[FloatArray, list[str] | None, list[str] | None, bool]


def _words_to_draw(
    instance: DocumentInstance,
    content: DocumentContent | None,
    *,
    draw_segment_bboxes: bool,
    draw_word_labels: bool,
) -> _DrawLayer | None:
    """Word-level XYXY bboxes, texts, labels, and normalization metadata."""
    ocr_ann: OCRAnnotation | None = instance.get_annotation_by_type(AnnotationType.ocr)
    elements = (
        content.elements
        if content is not None and content.elements is not None
        else ocr_ann
    )
    if elements is None:
        return None
    else:
        words = elements.at(OCRLevel.word)
        if words.bboxes is None or len(words) == 0:
            return None

        bboxes = (
            elements.segment_bboxes(OCRLevel.word)
            if draw_segment_bboxes
            else words.bboxes
        )
        if elements.bbox_mode == BoundingBoxMode.XYWH:
            bboxes = bboxes.copy()
            bboxes[:, 2:] += bboxes[:, :2]
        texts = words.texts.tolist() if words.texts is not None else None

        labels = None
        if draw_word_labels:
            ann = instance.get_annotation_by_type(AnnotationType.entity_labeling)
            if ann is not None:
                labels = ann.word_label_names

        return bboxes, texts, labels, elements.normalized


def _objects_to_draw(instance: DocumentInstance) -> _DrawLayer | None:
    """Object-detection/layout-analysis boxes, labelled by class name."""
    detection: ObjectDetectionAnnotation | None = instance.get_annotation_by_type(
        AnnotationType.object_detection
    )
    if detection is None or detection.bboxes is None or len(detection.bboxes) == 0:
        return None

    bboxes = detection.bboxes
    if detection.bbox_mode == BoundingBoxMode.XYWH:
        bboxes = bboxes.copy()
        bboxes[:, 2:] += bboxes[:, :2]
    labels = (
        detection.label_names.tolist() if detection.label_names is not None else None
    )
    return bboxes, labels, None, detection.normalized


def _render_document_image(
    instance: SinglePageDocumentInstance,
    *,
    draw_segment_bboxes: bool,
    draw_word_labels: bool,
    style: DrawStyle,
) -> PILImage.Image:
    image = instance.load().require_content().copy().convert("RGB")
    scale = np.array(
        [image.width, image.height, image.width, image.height], dtype=np.float64
    )
    layers = [
        _words_to_draw(
            instance,
            instance.content,
            draw_segment_bboxes=draw_segment_bboxes,
            draw_word_labels=draw_word_labels,
        ),
        _objects_to_draw(instance),
    ]
    for layer in layers:
        if layer is None:
            continue
        bboxes, texts, labels, normalized = layer
        if normalized:
            bboxes = bboxes * scale
        ImageDrawer().draw(
            image, bboxes.tolist(), texts=texts, labels=labels, style=style
        )
    return image


def _render_document_pdf_page(
    instance: SinglePageDocumentInstance,
    *,
    draw_segment_bboxes: bool,
    draw_word_labels: bool,
    style: DrawStyle,
) -> pymupdf.Document:
    assert (
        isinstance(instance.visual, PdfPage) and instance.visual.file_path is not None
    )
    pdf_bytes = ResourceLoader.for_uri(instance.visual.file_path).load_bytes()
    source_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out_doc = pymupdf.open()
    out_doc.insert_pdf(
        source_doc, from_page=instance.visual.page_id, to_page=instance.visual.page_id
    )
    page = out_doc[0]
    scale = np.array([
        page.rect.width,
        page.rect.height,
        page.rect.width,
        page.rect.height,
    ])

    layers = [
        _words_to_draw(
            instance,
            instance.content,
            draw_segment_bboxes=draw_segment_bboxes,
            draw_word_labels=draw_word_labels,
        ),
        _objects_to_draw(instance),
    ]
    for layer in layers:
        if layer is None:
            continue
        bboxes, texts, labels, normalized = layer
        if normalized:
            bboxes = bboxes * scale
        PdfDrawer().draw(page, bboxes, texts=texts, labels=labels, style=style)
    return out_doc


def _render_multi_page_document(
    instance: MultiPageDocumentInstance,
    *,
    draw_segment_bboxes: bool,
    draw_word_labels: bool,
    style: DrawStyle,
) -> pymupdf.Document:
    combined = pymupdf.open()
    for page in instance:
        if isinstance(page.visual, PdfPage):
            page_doc = _render_document_pdf_page(
                page,
                draw_segment_bboxes=draw_segment_bboxes,
                draw_word_labels=draw_word_labels,
                style=style,
            )
            combined.insert_pdf(page_doc)
        else:
            image = _render_document_image(
                page,
                draw_segment_bboxes=draw_segment_bboxes,
                draw_word_labels=draw_word_labels,
                style=style,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            combined.insert_file(pymupdf.Pixmap(buffer.getvalue()))
    return combined


def render_document_instance(
    instance: DocumentInstance,
    *,
    draw_segment_bboxes: bool = False,
    draw_word_labels: bool = True,
    style: DrawStyle = DEFAULT_STYLE,
) -> PILImage.Image | pymupdf.Document:
    """Draw `instance` and return the result in memory, without saving it --
    a PIL image for an image-sourced page, a pymupdf Document otherwise (one
    page for a PDF-sourced page, one page per page for a multi-page
    document)."""
    if isinstance(instance, MultiPageDocumentInstance):
        return _render_multi_page_document(
            instance,
            draw_segment_bboxes=draw_segment_bboxes,
            draw_word_labels=draw_word_labels,
            style=style,
        )

    assert isinstance(instance, SinglePageDocumentInstance)
    if isinstance(instance.visual, PdfPage):
        return _render_document_pdf_page(
            instance,
            draw_segment_bboxes=draw_segment_bboxes,
            draw_word_labels=draw_word_labels,
            style=style,
        )
    return _render_document_image(
        instance,
        draw_segment_bboxes=draw_segment_bboxes,
        draw_word_labels=draw_word_labels,
        style=style,
    )


def visualize_document_instance(
    instance: DocumentInstance,
    output_dir: str,
    *,
    draw_segment_bboxes: bool = False,
    draw_word_labels: bool = True,
    style: DrawStyle = DEFAULT_STYLE,
) -> Path:
    rendered = render_document_instance(
        instance,
        draw_segment_bboxes=draw_segment_bboxes,
        draw_word_labels=draw_word_labels,
        style=style,
    )
    extension = "png" if isinstance(rendered, PILImage.Image) else "pdf"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{output_name(instance)}.{extension}"
    logger.debug(f"Saving visualization for sample {instance.sample_id} to {path}")
    rendered.save(path)
    return path
