"""Correct way to log an exception raised inside a background thread.

The global exception hook (see 06_global_exception_hook_background_thread.py)
is a safety net for exceptions nobody caught - the preferred approach is
still to catch the exception where it happens and log it explicitly with
logger.exception(), just like in the main thread. Doing that here means
threading.excepthook never even fires, since the exception never escapes
the thread's target function.

Uses a temporary file so this example is fully self-contained.

Run: python usage/logger/07_manual_exception_logging_in_thread.py
"""

import logging
import tempfile
import threading
from pathlib import Path

from atria_core.logger import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)

with tempfile.TemporaryDirectory() as tmp_dir:
    log_file = Path(tmp_dir) / "atria_usage_thread_manual_exception.log"
    enable_file_logging(str(log_file))
    logger = get_logger(__name__)

    def _background_job() -> None:
        thread_logger = get_logger(f"{__name__}.worker")
        try:
            raise RuntimeError("Handled error in a background thread")
        except RuntimeError:
            thread_logger.exception("Background job failed")

    worker = threading.Thread(target=_background_job, name="worker-thread")
    worker.start()
    worker.join()

    logger.info("Main thread continues normally - the hook never had to fire")

    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    print(f"\n--- contents of {log_file.name} ---")
    print(log_file.read_text())
