# module1.py
# Somewhere in main application
import logging

from atria_logger.api import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)
logger = get_logger("atria.test")
logger = get_logger("atria.test1")
logger = get_logger("atria.test2")
logger = get_logger("atria.test3")
logger.info("root_logger1 INFO This goes to stdout and to file if attached")
logger.debug("root_logger2 DEBUGG This goes to stdout and to file if attached")
logger.debug("root_logger3 DEBUGG This goes to stdout and to file if attached")
set_atria_log_level(logging.DEBUG)
logger.debug("root_logger4 DEBUGG This goes to stdout and to file if attached")


enable_file_logging("./atria.log")
logger.info("This goes to file and stdout after attaching file handler")

logger.debug("This goes to file and stdout after attaching file handler")
