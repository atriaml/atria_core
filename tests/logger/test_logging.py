import logging
from pathlib import Path

import pytest

from atria_core.logger import get_logger, set_atria_log_level


# ----------------------------
# Test: Logger inherits root level
# ----------------------------
def test_logger_inherits_root_level() -> None:
    """Test that a module logger inherits its level from the root logger."""
    set_atria_log_level(logging.DEBUG)
    logger = get_logger("atria.module")
    # Child loggers have level 0, meaning they inherit root level
    assert logger.level == 0
    assert logger.name == "atria.module"
    assert logger.getEffectiveLevel() == logging.DEBUG


# ----------------------------
# Test: Debug messages propagate
# ----------------------------
def test_debug_message_propagation(caplog: pytest.LogCaptureFixture) -> None:
    """Test that DEBUG messages are captured when root level is DEBUG."""
    from atria_core.logger import set_atria_log_level

    set_atria_log_level(logging.DEBUG)
    logger = get_logger("atria.debugtest")
    # The atria root logger disables propagation to the real root logger (to
    # avoid duplicate output), so caplog's handler must be attached directly.
    logging.getLogger("atria").addHandler(caplog.handler)
    logger.debug("Debug message")
    messages = [rec.message for rec in caplog.records]
    assert "Debug message" in messages


# ----------------------------
# Test: Info messages propagate
# ----------------------------
def test_info_message_propagation(caplog: pytest.LogCaptureFixture) -> None:
    """Test that INFO messages propagate to the captured log."""
    from atria_core.logger import set_atria_log_level

    set_atria_log_level(logging.DEBUG)
    logger = get_logger("atria.infotest")
    logging.getLogger("atria").addHandler(caplog.handler)
    logger.info("Info message")
    messages = [rec.message for rec in caplog.records]
    assert "Info message" in messages


def test_file_logging(tmp_path: Path) -> None:
    """Test that Atria logging writes messages to a temporary file."""
    from atria_core.logger import enable_file_logging, set_atria_log_level

    # Prepare temp log file
    log_file = tmp_path / "atria_test.log"

    # Attach file to root logger
    enable_file_logging(str(log_file), level=logging.INFO)

    # Get a module logger (propagates to root)
    logger = get_logger("atria.filetest")

    # Ensure log level allows info messages
    set_atria_log_level(logging.INFO)

    # Log a test message
    test_msg = "File log test"
    logger.info(test_msg)

    # Flush all handlers to make sure content is written
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    # Verify file exists
    assert log_file.exists()

    # Verify content
    content = log_file.read_text()
    assert test_msg in content


# ----------------------------
# Test: Raising an exception alone is NOT logged
# ----------------------------
def test_raised_exception_not_logged_without_manual_handling(tmp_path: Path) -> None:
    """Test that merely raising/catching an exception does not write anything to the log file.

    Logging is not tied into Python's exception machinery automatically -
    a caught exception is only recorded if the code explicitly logs it
    (e.g. via `logger.exception(...)` or `logger.error(..., exc_info=True)`).
    """
    from atria_core.logger import enable_file_logging

    log_file = tmp_path / "atria_no_exception.log"
    enable_file_logging(str(log_file), level=logging.INFO)
    set_atria_log_level(logging.INFO)

    try:
        raise ValueError("boom - should not be logged")
    except ValueError:
        pass  # intentionally not logged

    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    content = log_file.read_text() if log_file.exists() else ""
    assert "boom - should not be logged" not in content
    assert "Traceback" not in content


# ----------------------------
# Test: logger.exception() logs the traceback to file
# ----------------------------
def test_logger_exception_writes_traceback_to_file(tmp_path: Path) -> None:
    """Test that explicitly calling `logger.exception()` inside an except block
    writes the error message and traceback to the log file.
    """
    from atria_core.logger import enable_file_logging

    log_file = tmp_path / "atria_exception.log"
    enable_file_logging(str(log_file), level=logging.INFO)
    set_atria_log_level(logging.INFO)

    logger = get_logger("atria.exceptiontest.handled")

    try:
        raise ValueError("boom - should be logged")
    except ValueError:
        logger.exception("Something went wrong")

    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    content = log_file.read_text()
    assert "Something went wrong" in content
    assert "Traceback (most recent call last)" in content
    assert "ValueError: boom - should be logged" in content
