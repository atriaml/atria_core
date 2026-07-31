"""Basic logging: get a module logger, set the log level, log at each level.

Run: python usage/02_basic_logging.py
"""

import logging

from atria_logger import get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)

logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
