"""Attach a file handler so logs are written to disk as well as the console.

Run: python usage/03_file_logging.py
Then check: /tmp/atria_usage_file_logging.log
"""

import logging

from atria_logger import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)
enable_file_logging("/tmp/atria_usage_file_logging.log")

logger = get_logger(__name__)
logger.info("This message is written to both the console and the log file")
