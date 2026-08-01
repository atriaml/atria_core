"""Atria Core - shared core utilities for Atria projects.

Submodules (e.g. `logger`) are attached lazily so importing `atria_core`
doesn't eagerly import every submodule's dependencies. Add new submodules to
the `submodules` list below as they're introduced.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "logger",
        "types",
        "transforms",
        "extractors",
        "visualizers",
        "serialization",
    ],
)
