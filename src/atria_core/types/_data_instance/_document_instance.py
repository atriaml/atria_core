from __future__ import annotations

from typing import Self

from pydantic import model_validator

from atria_core.logger import get_logger
from atria_core.types._base._data_model import _load_any
from atria_core.types._base._ops._base_ops import StandardOps
from atria_core.types._data_instance._base import BaseDataInstance
from atria_core.types._data_instance._visualizers._document import DocumentVisualizer
from atria_core.types._generic._annotations import (
    AnnotationType,
    QuestionAnsweringAnnotation,
)
from atria_core.types._generic._doc_content import DocumentContent
from atria_core.types._generic._image import Image
from atria_core.types._generic._ocr import OCR
from atria_core.types._generic._pdf import PDF
from atria_core.types._generic._qa_pair import QAPair
from atria_core.types._ocr_engines import OCREngineConfigType
from atria_core.types._ocr_engines._tesseract import TesseractOCREngineConfig
from atria_core.types._utilities._ocr_processors._base import OCRProcessor

logger = get_logger(__name__)

_DEFAULT_OCR_CONFIG = TesseractOCREngineConfig(lang="eng", psm=3, oem=3)


def has_answer(qa_pair: QAPair):
    for answer_span in qa_pair.answer_spans:
        if answer_span.start != -1 and answer_span.end != -1:
            return True
    return False


class DocumentInstance(BaseDataInstance):
    page_id: int | None = None
    pdf: PDF | None = None
    image: Image | None = None
    ocr: OCR | None = None
    content: DocumentContent | None = None

    @property
    def viz(self) -> DocumentVisualizer:
        return DocumentVisualizer(instance=self)

    @property
    def ops(self) -> DocumentOps:
        return DocumentOps(self)

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        # Ensure we have either image or PDF
        if self.image is None and self.pdf is None:
            raise ValueError("Either image or pdf must be provided")
        return self

    def load(
        self,
    ) -> Self:
        loaded_fields = {}
        for name in self.__class__.model_fields:
            loaded_fields[name] = _load_any(getattr(self, name))
        if (
            loaded_fields["ocr"] is not None
            and loaded_fields["ocr"].content is not None
            and loaded_fields["content"] is None
        ):
            loaded_fields["content"] = OCRProcessor.parse(
                raw_ocr=loaded_fields["ocr"].content, ocr_type=loaded_fields["ocr"].type
            )

        new_instance = self.model_copy(update=loaded_fields)
        return new_instance


class DocumentOps(StandardOps[DocumentInstance]):
    @property
    def doc(self) -> DocumentInstance:
        return self.model

    def load_image_from_pdf(self, page_number: int = 0) -> DocumentInstance:
        if self.model.image is not None:
            logger.warning("Image already exists. Skipping load from PDF.")
            return self.model

        if self.model.pdf is None or self.model.pdf.file_path is None:
            raise ValueError("PDF file path must be set to load image from PDF.")

        pdf = self.model.pdf.load()
        pil_image = pdf.get_page(page_number)

        return self.model.model_copy(
            update={
                "image": Image(file_path=self.model.pdf.file_path, content=pil_image)
            }
        )

    def load_content_from_pdf(self, page_number: int = 0) -> DocumentInstance:
        content = self.model.content
        if content is not None:
            logger.warning("Document content already exists. Skipping load from PDF.")
            return self.model

        if self.model.pdf is None or self.model.pdf.file_path is None:
            raise ValueError("PDF file path must be set to load content from PDF.")

        pdf = self.model.pdf.load()
        text_elements = pdf.extract_text_elements(page_number=page_number)
        return self.model.model_copy(
            update={"content": DocumentContent(text_elements=text_elements)}
        )

    def load_content_from_image(
        self,
        config: OCREngineConfigType = _DEFAULT_OCR_CONFIG,
    ) -> DocumentInstance:
        if self.model.image is None:
            raise ValueError("Image must be set to load OCR from image.")

        ocr_engine = config.build()
        text_elements = ocr_engine.extract_text_elements(self.model.image)
        return self.model.model_copy(
            update={
                "content": DocumentContent(text_elements=text_elements),
            }
        )

    def get_instances_per_qa_pair(
        self, ignore_no_answer_qa_pair: bool = False
    ) -> list[DocumentInstance]:
        assert self.model.has_annotation_type(
            annotation_type=AnnotationType.question_answering
        ), (
            f"{self.model.__class__.__name__} {self.model.sample_id} does not have "
            f"QuestionAnsweringAnnotation."
        )
        qa_annotation = self.model.get_annotation_by_type(
            annotation_type=AnnotationType.question_answering
        )
        logger.debug(
            f"Found `{len(qa_annotation.qa_pairs)}` qa pairs in sample `{self.model.sample_id}`"
        )
        document_instance_per_qa_pair = []
        for qa_pair in qa_annotation.qa_pairs:
            assert len(qa_pair.answer_spans) > 0, (
                f"QA Pair {qa_pair.id} has no answer spans."
            )
            if ignore_no_answer_qa_pair:
                if not has_answer(qa_pair):
                    logger.debug(
                        f"Skipping QA Pair with id {qa_pair.id} due to no answer spans."
                    )
                    continue

            # create a new DocumentInstance for this QA pair
            qa_document_instance = self.model.model_copy(
                update={
                    "sample_id": f"{self.model.sample_id}_qa_{qa_pair.id}",
                    "annotations": [QuestionAnsweringAnnotation(qa_pairs=[qa_pair])],
                }
            )
            document_instance_per_qa_pair.append(qa_document_instance)
        return document_instance_per_qa_pair

    def get_qa_pairs(self) -> list[QAPair]:
        assert self.model.has_annotation_type(
            annotation_type=AnnotationType.question_answering
        ), (
            f"{self.model.__class__.__name__} {self.model.sample_id} does not have "
            f"QuestionAnsweringAnnotation."
        )
        qa_annotation = self.model.get_annotation_by_type(
            annotation_type=AnnotationType.question_answering
        )
        return qa_annotation.qa_pairs
