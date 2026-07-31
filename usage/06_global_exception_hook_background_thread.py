"""Fail-safe logging for exceptions that escape a background thread.

threading.excepthook fires independently of the main thread's sys.excepthook -
it's a separate mechanism, only triggered for exceptions that escape a
Thread's target function. The crash only kills that thread; the main thread
keeps running.

Run: python usage/06_global_exception_hook_background_thread.py
Then check: /tmp/atria_usage_background_thread_hook.log
"""

import logging
import threading

from atria_logger import (
    enable_file_logging,
    get_logger,
    install_global_exception_hook,
    set_atria_log_level,
)

set_atria_log_level(logging.DEBUG)
enable_file_logging("/tmp/atria_usage_background_thread_hook.log")
logger = get_logger(__name__)

install_global_exception_hook()


def _background_job() -> None:
    raise RuntimeError("Uncaught error in a background thread")


worker = threading.Thread(target=_background_job, name="worker-thread")
worker.start()
worker.join()

logger.info("Main thread is still alive after the worker thread crashed")
