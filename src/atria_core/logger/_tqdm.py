"""A file-like adapter for routing tqdm's own output into a logger.

tqdm renders by writing plain text to a stream, not by emitting
`logging.LogRecord`s, so a file handler attached via `enable_file_logging`
never sees a plain `tqdm`'s updates on its own. Passing `TqdmLogger` as
tqdm's `file` argument gets those updates into a logger -- and from there
into any handler attached to it -- without changing tqdm's own behavior at
all.
"""

import logging


class TqdmLogger:
    """File-like adapter that forwards non-empty `write()` calls to a logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def write(self, message: str) -> None:
        stripped = message.strip()
        if stripped:
            self._logger.info(stripped)

    def flush(self) -> None:
        pass
