"""Correct way to log an exception you catch.

Call logger.exception() (or logger.error(..., exc_info=True)) from inside the
except block. Raising and catching an exception alone writes nothing to the
log - logging isn't tied into Python's exception machinery automatically.

Run: python usage/logger/04_manual_exception_logging.py
Then check: /tmp/atria_usage_manual_exception.log
"""

import logging

from atria_core.logger import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)
enable_file_logging("/tmp/atria_usage_manual_exception.log")

logger = get_logger(__name__)

try:
    result = 1 / 0
    logger.debug("Unreachable: %s", result)
except ZeroDivisionError:
    logger.exception("Handled division error")
