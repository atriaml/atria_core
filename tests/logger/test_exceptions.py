"""Unit tests for the global fail-safe exception logging hooks.

These hooks are installed automatically as soon as `atria_core.logger` is
configured (see `RootLoggerAdapter.configure` in `_root.py`), so none of
these tests call `install_global_exception_hook()` themselves.
"""

import logging
import sys
import threading
from pathlib import Path

import pytest

from atria_core.logger import enable_file_logging, set_atria_log_level


def _flush_atria_handlers() -> None:
    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()


def test_global_exception_hooks_are_installed_by_default() -> None:
    """Test that importing/using the logger installs the hooks without an
    explicit `install_global_exception_hook()` call.
    """
    assert sys.excepthook is not sys.__excepthook__
    assert threading.excepthook is not threading.__excepthook__


def test_uncaught_main_thread_exception_is_logged(tmp_path: Path) -> None:
    """Test that an exception which escapes every try/except in the main
    thread gets logged via the installed `sys.excepthook`.
    """
    log_file = tmp_path / "atria_uncaught_main.log"
    enable_file_logging(str(log_file), level=logging.INFO)
    set_atria_log_level(logging.INFO)

    try:
        raise RuntimeError("uncaught in main thread")
    except RuntimeError:
        exc_info = sys.exc_info()

    # This is what the interpreter itself does when an exception propagates
    # all the way up without being caught - invoke sys.excepthook with the
    # exception details.
    sys.excepthook(*exc_info)

    _flush_atria_handlers()

    content = log_file.read_text()
    assert "Unhandled exception" in content
    assert "RuntimeError: uncaught in main thread" in content


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_uncaught_background_thread_exception_is_logged(tmp_path: Path) -> None:
    """Test that an exception which escapes a background thread's target
    function gets logged via the installed `threading.excepthook`, and does
    not affect the main thread.
    """
    log_file = tmp_path / "atria_uncaught_thread.log"
    enable_file_logging(str(log_file), level=logging.INFO)
    set_atria_log_level(logging.INFO)

    def _boom() -> None:
        raise RuntimeError("uncaught in background thread")

    worker = threading.Thread(target=_boom, name="test-worker-thread")
    worker.start()
    worker.join()

    _flush_atria_handlers()

    content = log_file.read_text()
    assert "Unhandled exception in thread test-worker-thread" in content
    assert "RuntimeError: uncaught in background thread" in content
