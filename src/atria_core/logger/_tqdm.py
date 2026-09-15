"""A tqdm subclass that periodically logs progress-bar state to the log file.

tqdm never emits `logging.LogRecord`s, so a `logging.FileHandler` attached via
`enable_file_logging` never sees progress-bar text on its own. Mirroring every
in-place console update into a plain file would turn each bar into hundreds of
near-duplicate lines, but logging only on completion leaves nothing in the log
for a bar that is still running or that never finishes (e.g. the process is
killed mid-loop). `FileLoggingTqdm` instead logs the bar's current rendered
state through the normal logger at most once every `log_interval` seconds,
plus a final line on `close()`.
"""

import time
from typing import Any

from tqdm.auto import tqdm as _auto_tqdm

from ._root import get_root_adapter

DEFAULT_LOG_INTERVAL_SECONDS = 30.0


class FileLoggingTqdm(_auto_tqdm):  # type: ignore[misc]
    """`tqdm.auto.tqdm` that also logs its state at a fixed time interval.

    Behaves identically to `tqdm.auto.tqdm` for console display. In addition,
    at most once every `log_interval` seconds (and once more when the bar
    closes), it emits an INFO log record, through the Atria root logger, with
    its current rendered state -- so a run's attached log file gets periodic
    progress snapshots for long-running or interrupted loops, not just a
    summary line for loops that finish.

    Args:
        log_interval: Minimum seconds between progress log records. Defaults
            to `DEFAULT_LOG_INTERVAL_SECONDS`.
    """

    def __init__(
        self,
        *args: Any,
        log_interval: float = DEFAULT_LOG_INTERVAL_SECONDS,
        **kwargs: Any,
    ) -> None:
        self._log_interval = log_interval
        self._last_logged_at: float | None = None
        super().__init__(*args, **kwargs)

    def display(self, msg: str | None = None, pos: int | None = None) -> bool | None:
        result = super().display(msg=msg, pos=pos)
        self._maybe_log_progress()
        return result

    def close(self, *args: Any, **kwargs: Any) -> None:
        already_disabled = self.disable
        rendered_line = str(self)
        super().close(*args, **kwargs)
        if not already_disabled:
            get_root_adapter().logger.info(rendered_line)

    def _maybe_log_progress(self) -> None:
        if self.disable:
            return

        now = time.monotonic()
        if (
            self._last_logged_at is not None
            and now - self._last_logged_at < self._log_interval
        ):
            return

        self._last_logged_at = now
        get_root_adapter().logger.info(str(self))
