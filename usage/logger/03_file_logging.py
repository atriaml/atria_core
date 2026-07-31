"""Attach a file handler so logs are written to disk as well as the console.

Uses a temporary file so this example is fully self-contained: it prints the
file's contents at the end and the temp directory is removed automatically.

Run: python usage/logger/03_file_logging.py
"""

import logging
import tempfile
from pathlib import Path

from atria_core.logger import enable_file_logging, get_logger, set_atria_log_level

set_atria_log_level(logging.DEBUG)

with tempfile.TemporaryDirectory() as tmp_dir:
    log_file = Path(tmp_dir) / "atria_usage_file_logging.log"
    enable_file_logging(str(log_file))

    logger = get_logger(__name__)
    logger.info("This message is written to both the console and the log file")

    for handler in logging.getLogger("atria").handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    print(f"\n--- contents of {log_file.name} ---")
    print(log_file.read_text())
