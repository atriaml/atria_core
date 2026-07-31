from abc import ABC, abstractmethod
from typing import Any

from atria_core.logger import get_logger
from atria_core.types import TextElement

logger = get_logger(__name__)


class BaseOCREngine(ABC):
    @abstractmethod
    def extract_text_elements(self, image: Any) -> list[TextElement]:
        pass
