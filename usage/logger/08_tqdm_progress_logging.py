"""Log a plain tqdm progress bar's rendered state to a file.

`atria_core.logger.TqdmLogger` is a file-like adapter that forwards tqdm's
`write()` calls to a logger, so passing it as `file=` gets the bar's rendered
lines into the log file without changing tqdm's own behavior at all.

Run: python usage/logger/08_tqdm_progress_logging.py
"""

import logging
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

from atria_core.logger import (
    TqdmLogger,
    enable_file_logging,
    get_logger,
    set_atria_log_level,
)

set_atria_log_level(logging.DEBUG)

with tempfile.TemporaryDirectory() as tmp_dir:
    log_file = Path(tmp_dir) / "atria_usage_tqdm_logging.log"
    enable_file_logging(str(log_file))
    logger = get_logger(__name__)

    for _ in tqdm(
        range(5), desc="processing", file=TqdmLogger(logger), mininterval=0
    ):
        time.sleep(0.1)

    for handler in logging.getLogger("atria").handlers:
        handler.flush()

    print(f"\n--- contents of {log_file.name} ---")
    print(log_file.read_text())
