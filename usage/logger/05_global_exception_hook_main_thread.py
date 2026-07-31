"""Fail-safe logging for exceptions that escape every try/except in the main
thread.

The global exception hook is installed automatically the first time
`atria_core.logger` is configured - no setup call needed. sys.excepthook is
called by the interpreter only once an exception has propagated all the way
up with nothing catching it; our hook logs it at CRITICAL before the default
traceback printing runs and the process exits - this script ends by raising
on purpose so you can see that.

Uses a temporary file so this example is self-contained. Since the crash
skips past any code written after the raise, printing the file's contents
and cleaning up the temp dir is done from an atexit hook - atexit callbacks
still run during interpreter shutdown even after an unhandled exception, by
which point sys.excepthook has already logged to the file.

Run: python usage/logger/05_global_exception_hook_main_thread.py
"""

import atexit
import logging
import tempfile
from pathlib import Path

from atria_core.logger import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)

_tmp_dir = tempfile.TemporaryDirectory()
log_file = Path(_tmp_dir.name) / "atria_usage_main_thread_hook.log"
enable_file_logging(str(log_file))

logger = get_logger(__name__)


@atexit.register
def _print_and_cleanup() -> None:
    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()
    print(f"\n--- contents of {log_file.name} ---")
    print(log_file.read_text())
    _tmp_dir.cleanup()


logger.info("About to raise an exception that nothing catches...")
raise RuntimeError("Uncaught error in the main thread")
