from atria_core.types._common import OCRType
from atria_core.types._generic._doc_content import DocumentContent


class OCRProcessor:
    @staticmethod
    def parse(raw_ocr: str, ocr_type: OCRType) -> DocumentContent:
        if ocr_type == OCRType.tesseract:
            from atria_core.types._utilities._ocr_processors._hocr_processor import (
                HOCRProcessor,
            )

            return HOCRProcessor.parse(raw_ocr)
        else:
            raise ValueError(f"Unsupported OCR type: {ocr_type}")
