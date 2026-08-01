from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf

from atria_core.logger import get_logger
from atria_core.types import (
    AnnotationType,
    DocumentContent,
    DocumentInstance,
    MultiPageDocument,
    OCRLevel,
    ResourceLoader,
    SinglePageDocument,
)
from atria_core.visualizers._drawers._image import ImageDrawer
from atria_core.visualizers._drawers._pdf import PdfDrawer
from atria_core.visualizers._drawers._style import DEFAULT_STYLE, DrawStyle
from atria_core.visualizers.functional._visualize import output_name

logger = get_logger(__name__)


def _words_to_draw(
    instance: DocumentInstance,
    content: DocumentContent | None,
    *,
    draw_segment_bboxes: bool,
    draw_word_labels: bool,
) -> tuple[np.ndarray, list[str] | None, list[str] | None] | None:
    """Word-level bboxes (still normalized [0,1]), texts, and optional
    entity-labeling labels -- shared prep for both the image and PDF
    drawing targets, which each scale these to their own coordinate space."""
    if content is None or content.elements is None:
        return None

    words = content.elements.at(OCRLevel.word)
    if words.bboxes is None or len(words) == 0:
        return None

    bboxes = (
        content.elements.segment_bboxes(OCRLevel.word) if draw_segment_bboxes else words.bboxes
    )
    texts = words.texts.tolist() if words.texts is not None else None

    labels = None
    if draw_word_labels:
        ann = instance.get_annotation_by_type(AnnotationType.entity_labeling)
        if ann is not None:
            labels = ann.label_names

    return bboxes, texts, labels


def _visualize_document_image(
    instance: DocumentInstance,
    document: SinglePageDocument,
    output_dir: str,
    *,
    draw_segment_bboxes: bool,
    draw_word_labels: bool,
    style: DrawStyle,
) -> Path:
    image = document.image.copy().convert("RGB")
    prepared = _words_to_draw(
        instance,
        document.content,
        draw_segment_bboxes=draw_segment_bboxes,
        draw_word_labels=draw_word_labels,
    )
    if prepared is not None:
        bboxes, texts, labels = prepared
        scale = np.array([image.width, image.height, image.width, image.height])
        ImageDrawer().draw(image, bboxes * scale, texts=texts, labels=labels, style=style)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{output_name(instance)}.png"
    logger.debug(f"Saving visualization for sample {instance.sample_id} to {path}")
    image.save(path)
    return path


def _visualize_document_pdf_page(
    instance: DocumentInstance,
    document: SinglePageDocument,
    output_dir: str,
    *,
    draw_segment_bboxes: bool,
    draw_word_labels: bool,
    style: DrawStyle,
) -> Path:
    assert document.source_path is not None and document.page_id is not None
    pdf_bytes = ResourceLoader.for_uri(document.source_path).load_bytes()
    source_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
    out_doc = pymupdf.open()  # type: ignore[no-untyped-call]
    out_doc.insert_pdf(  # type: ignore[no-untyped-call]
        source_doc, from_page=document.page_id, to_page=document.page_id
    )
    page = out_doc[0]

    prepared = _words_to_draw(
        instance,
        document.content,
        draw_segment_bboxes=draw_segment_bboxes,
        draw_word_labels=draw_word_labels,
    )
    if prepared is not None:
        bboxes, texts, labels = prepared
        scale = np.array(
            [page.rect.width, page.rect.height, page.rect.width, page.rect.height]
        )
        PdfDrawer().draw(page, bboxes * scale, texts=texts, labels=labels, style=style)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{output_name(instance)}.pdf"
    logger.debug(f"Saving visualization for sample {instance.sample_id} to {path}")
    out_doc.save(path)  # type: ignore[no-untyped-call]
    return path


def _visualize_multi_page_document(
    instance: DocumentInstance,
    document: MultiPageDocument,
    output_dir: str,
) -> Path:
    # MultiPageDocument carries no per-page DocumentContent -- there's
    # nothing page-specific to draw yet, so this just re-saves a valid
    # multi-page PDF rather than raising. See _drawers/ for the drawing
    # infrastructure a future per-page-content shape would plug into.
    pdf_bytes = ResourceLoader.for_uri(document.source_path).load_bytes()
    source_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{output_name(instance)}.pdf"
    logger.debug(f"Saving visualization for sample {instance.sample_id} to {path}")
    source_doc.save(path)  # type: ignore[no-untyped-call]
    return path


def visualize_document_instance(
    instance: DocumentInstance,
    output_dir: str,
    *,
    draw_segment_bboxes: bool = False,
    draw_word_labels: bool = True,
    style: DrawStyle = DEFAULT_STYLE,
) -> Path:
    document = instance.document

    if isinstance(document, MultiPageDocument):
        return _visualize_multi_page_document(instance, document, output_dir)

    is_pdf_page = (
        document.source_path is not None
        and document.source_path.lower().endswith(".pdf")
        and document.page_id is not None
    )
    if is_pdf_page:
        return _visualize_document_pdf_page(
            instance,
            document,
            output_dir,
            draw_segment_bboxes=draw_segment_bboxes,
            draw_word_labels=draw_word_labels,
            style=style,
        )
    return _visualize_document_image(
        instance,
        document,
        output_dir,
        draw_segment_bboxes=draw_segment_bboxes,
        draw_word_labels=draw_word_labels,
        style=style,
    )
