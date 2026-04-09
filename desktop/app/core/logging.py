"""Application-wide logging configuration.

Sets up structured JSON logging via Python's dictConfig, using the log level
specified in application settings.
"""

from __future__ import annotations

import logging.config
from typing import Any

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Apply the application-wide logging configuration based on settings."""
    log_level = settings.log_level.upper()
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": (
                    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
                    '"message":"%(message)s"}'
                ),
            },
            "plain": {
                "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            },
        },
        "handlers": {
            "default": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "json",
            }
        },
        "root": {"handlers": ["default"], "level": log_level},
    }
    logging.config.dictConfig(config)
