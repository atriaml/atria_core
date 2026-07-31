"""Logging Utilities Module.

Utilities to configure and manage loggers in the Atria library. Includes:
- attaching file handlers
- adding stream handlers
- enabling colored logging
"""

import logging
from pathlib import Path
from typing import Any, TextIO

import coloredlogs  # type: ignore[import-untyped]


def _next_available_path(path: Path) -> Path:
    """Return `path`, or if it already exists, the same path with an
    incrementing numeric suffix appended (e.g. app.log -> app.log.1 ->
    app.log.2 ...) until one that doesn't exist is found.

    This keeps re-runs that log to the same fixed path from silently
    appending to (or being confused with) a previous run's log file.
    """
    if not path.exists():
        return path

    counter = 1
    candidate = path.with_name(f"{path.name}.{counter}")
    while candidate.exists():
        counter += 1
        candidate = path.with_name(f"{path.name}.{counter}")
    return candidate


def _enable_colored_logging(
    logger: logging.Logger,
    log_level: int,
    styles: dict[str, dict[str, Any]],
    log_format: str,
) -> None:
    """Enable colored logging for the specified logger.

    Args:
        logger: The logger to configure.
        log_level: The logging level (e.g., DEBUG, INFO).
        styles: Color styles for each logging level.
        log_format: Format string for log messages.
    """
    coloredlogs.install(
        level=log_level, logger=logger, fmt=log_format, level_styles=styles
    )


def _attach_file_handler(
    logger: logging.Logger,
    log_file_path: str,
    log_level: int = logging.INFO,
    log_format: str = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
) -> logging.FileHandler:
    """Attach a FileHandler to a logger.

    Args:
        logger: Logger to attach the handler to.
        log_file_path: Path to log file.
        log_level: Logging level (default INFO).
        log_format: Format string for log messages (default standard format).

    Returns:
        The attached FileHandler instance. Its `baseFilename` may differ from
        `log_file_path` if that path already existed - see `_next_available_path`.
    """
    resolved_path = _next_available_path(Path(log_file_path))
    handler = logging.FileHandler(resolved_path)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    return handler


def _attach_stream_handler(
    logger: logging.Logger,
    log_stream: TextIO,
    log_level: int = logging.INFO,
    log_format: str = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
) -> logging.StreamHandler:  # type: ignore[type-arg]
    """Attach a StreamHandler to a logger.

    Args:
        logger: Logger to attach the handler to.
        log_stream: Stream to log messages to (e.g., sys.stdout).
        log_level: Logging level (default INFO).
        log_format: Format string for log messages (default standard format).

    Returns:
        The attached StreamHandler instance.
    """
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    return handler


def _reset_logger(logger: logging.Logger) -> None:
    # Remove all handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # Remove all filters
    for filter_ in list(logger.filters):
        logger.removeFilter(filter_)

    # Reset level
    logger.setLevel(logging.NOTSET)

    # Optional: clear propagate flag
    logger.propagate = True
