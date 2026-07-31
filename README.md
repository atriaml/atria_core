# Atria Core

Atria Core is a collection of shared core utilities for Atria projects. Submodules
are attached lazily, so importing `atria_core` only pulls in the pieces you use.

## Submodules

### `atria_core.logger`

A lightweight and flexible Python logging library designed to simplify
logging and monitoring application behavior. It provides an easy-to-use interface
with support for colored logs, distributed environments, and file logging.

- Centralized root logger for your library/application.
- Colored logs using `coloredlogs` for better readability.
- Optional file logging for different log levels.
- Environment variable support for default log level and process rank.
- Propagation-friendly module-level loggers.

### `atria_core.types`

Structured, immutable pydantic data models for document/OCR datasets
(images, PDFs, bounding boxes, annotations, OCR results, document and image
instances) along with a small OCR engine abstraction (Tesseract included).

---

## Installation

```bash
pip install atria_core
```

# Usage 
```bash
import logging

from atria_core.logger import enable_file_logging, get_logger, set_atria_log_level

# Set global log level
set_atria_log_level(logging.DEBUG)

# Attach log file
enable_file_logging("app.log")

# Get a module-level logger
logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")

```
