"""Fail-safe logging for exceptions that escape a background thread.

The global exception hook is installed automatically the first time
`atria_core.logger` is configured - no setup call needed. threading.excepthook
fires independently of the main thread's sys.excepthook - it's a separate
mechanism, only triggered for exceptions that escape a Thread's target
function. The crash only kills that thread; the main thread keeps running.

Uses a temporary file so this example is fully self-contained.

Run: python usage/logger/06_global_exception_hook_background_thread.py
"""

import logging
import tempfile
import threading
from pathlib import Path

from atria_core.logger import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)

with tempfile.TemporaryDirectory() as tmp_dir:
    log_file = Path(tmp_dir) / "atria_usage_background_thread_hook.log"
    enable_file_logging(str(log_file))
    logger = get_logger(__name__)

    def _background_job() -> None:
        raise RuntimeError("Uncaught error in a background thread")

    worker = threading.Thread(target=_background_job, name="worker-thread")
    worker.start()
    worker.join()

    logger.info("Main thread is still alive after the worker thread crashed")

    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    print(f"\n--- contents of {log_file.name} ---")
    print(log_file.read_text())
