"""Correct way to log an exception you catch.

Call logger.exception() (or logger.error(..., exc_info=True)) from inside the
except block. Raising and catching an exception alone writes nothing to the
log - logging isn't tied into Python's exception machinery automatically.

Uses a temporary file so this example is fully self-contained.

Run: python usage/logger/04_manual_exception_logging.py
"""

import logging
import tempfile
from pathlib import Path

from atria_core.logger import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)

with tempfile.TemporaryDirectory() as tmp_dir:
    log_file = Path(tmp_dir) / "atria_usage_manual_exception.log"
    enable_file_logging(str(log_file))

    logger = get_logger(__name__)

    try:
        result = 1 / 0
        logger.debug("Unreachable: %s", result)
    except ZeroDivisionError:
        logger.exception("Handled division error")

    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    print(f"\n--- contents of {log_file.name} ---")
    print(log_file.read_text())
