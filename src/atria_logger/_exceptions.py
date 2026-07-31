"""Fail-safe logging hooks for exceptions that escape every try/except block.

These hooks are a safety net for exceptions nobody caught - they do not
replace explicit `logger.exception()` calls at handled call sites, and they
cannot see exceptions that were caught and deliberately swallowed.
"""

import sys
import threading
from types import TracebackType

from ._api import get_logger

_logger = get_logger("atria.uncaught")

_installed = False
_original_excepthook = sys.excepthook
_original_threading_excepthook = threading.excepthook


def _log_uncaught_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> None:
    if not issubclass(exc_type, KeyboardInterrupt):
        _logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
    _original_excepthook(exc_type, exc_value, exc_tb)


def _log_uncaught_thread_exception(args: threading.ExceptHookArgs) -> None:
    thread_name = args.thread.name if args.thread is not None else "unknown"
    _logger.critical(
        "Unhandled exception in thread %s",
        thread_name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    _original_threading_excepthook(args)


def install_global_exception_hook() -> None:
    """Install fail-safe logging for exceptions that escape every try/except.

    Once installed, any exception that propagates all the way up in the main
    thread, or out of a `threading.Thread`'s target function, is logged at
    CRITICAL level before the interpreter's default handling (printing to
    stderr and unwinding) runs. Idempotent - calling it more than once has no
    additional effect.

    Does NOT catch exceptions that are caught and swallowed by application
    code, nor exceptions raised inside `concurrent.futures` worker functions
    that are never retrieved via `future.result()`, nor `asyncio` task
    exceptions - those each have their own separate top-level exception
    handling and need to be logged explicitly at the call site.
    """
    global _installed
    if _installed:
        return
    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_uncaught_thread_exception
    _installed = True
