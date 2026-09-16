# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._drawers._base import BboxDrawer as BboxDrawer
    from ._drawers._image import ImageDrawer as ImageDrawer
    from ._drawers._pdf import PdfDrawer as PdfDrawer
    from ._drawers._style import DrawStyle as DrawStyle
    from .functional import output_name as output_name, visualize as visualize
    from .functional import (
        render_document_instance as render_document_instance,
        visualize_document_instance as visualize_document_instance,
        visualize_image_instance as visualize_image_instance,
    )

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_drawers._base": ["BboxDrawer"],
        "_drawers._image": ["ImageDrawer"],
        "_drawers._pdf": ["PdfDrawer"],
        "_drawers._style": ["DrawStyle"],
        "functional": [
            "output_name",
            "render_document_instance",
            "visualize",
            "visualize_document_instance",
            "visualize_image_instance",
        ],
    },
)
