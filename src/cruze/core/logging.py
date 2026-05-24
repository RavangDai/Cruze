"""
Structured logging setup with rotation.

Call configure_logging(cfg) once at startup from orchestrator.py.
Outputs JSON lines in production (structured=True) or coloured text for dev.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import pathlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cruze.core.config import LoggingConfig


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj)


class _PrettyFormatter(logging.Formatter):
    _COLOURS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        name = record.name.replace("cruze.", "")
        msg = record.getMessage()
        line = f"{colour}[{ts}] {record.levelname:<8}{self._RESET} {name}: {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(cfg: "LoggingConfig") -> None:
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console handler — always pretty in dev, JSON in structured mode.
    console = logging.StreamHandler()
    console.setFormatter(_PrettyFormatter() if not cfg.structured else _JsonFormatter())
    root.addHandler(console)

    # Rotating file handler — always JSON for machine parsing.
    log_dir = pathlib.Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "cruze.log",
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
    )
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging initialised at level=%s structured=%s", cfg.level, cfg.structured
    )
