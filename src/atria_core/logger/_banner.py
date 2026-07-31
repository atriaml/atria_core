"""ASCII art startup banner for the Atria logger."""

import logging
import os
import platform
import socket
from dataclasses import dataclass, fields
from functools import cache
from importlib import metadata

from pyfiglet import Figlet

from ._api import get_logger

logger = get_logger("atria")

_APP_NAME = "ATRIA"
_BANNER_FONT = "slant"
_DISTRIBUTION_NAME = "atria_core"

_figlet = Figlet(font=_BANNER_FONT)


def _atria_version() -> str:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class EnvInfo:
    """Runtime info shown in the startup banner. Add new fields here as needed."""

    version: str
    python_version: str
    hostname: str
    pid: int
    rank: str
    log_level: str

    def format(self) -> str:
        """Render as a single "key=value ..." line."""
        return " ".join(f"{field.name}={getattr(self, field.name)}" for field in fields(self))


@cache
def _collect_env_info() -> EnvInfo:
    return EnvInfo(
        version=_atria_version(),
        python_version=platform.python_version(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        rank=os.environ.get("RANK", "0"),
        log_level=logging.getLevelName(logger.getEffectiveLevel()),
    )


def log_banner() -> None:
    """Log the ATRIA startup banner along with basic runtime information.

    Emits the ASCII art name followed by a summary line (package version,
    Python version, hostname, PID, distributed rank, effective log level) -
    the kind of one-time startup banner frameworks like Celery or Ray print.
    """
    banner = _figlet.renderText(_APP_NAME).rstrip("\n")
    logger.info("\n%s", banner)
    logger.info(_collect_env_info().format())
