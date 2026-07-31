"""Fail-safe logging for exceptions that escape every try/except in the main
thread.

sys.excepthook is called by the interpreter only once an exception has
propagated all the way up with nothing catching it. install_global_exception_hook()
logs it at CRITICAL before the default traceback printing runs and the
process exits - this script ends by raising on purpose so you can see that.

Run: python usage/05_global_exception_hook_main_thread.py
Then check: /tmp/atria_usage_main_thread_hook.log
"""

import logging

from atria_logger import (
    enable_file_logging,
    get_logger,
    install_global_exception_hook,
    set_atria_log_level,
)

set_atria_log_level(logging.DEBUG)
enable_file_logging("/tmp/atria_usage_main_thread_hook.log")
logger = get_logger(__name__)

install_global_exception_hook()

logger.info("About to raise an exception that nothing catches...")
raise RuntimeError("Uncaught error in the main thread")
