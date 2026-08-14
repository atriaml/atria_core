from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RegistryStore:
    """Reads/writes a JSON-serializable mapping to a file. Has no knowledge
    of what it stores -- callers decide what data to pass in and how to
    read it back (see Registry.to_dict())."""

    @staticmethod
    def dump(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
