"""
Centralized logging configuration.

This module configures application-wide logging for both
console and file output.
"""
"""
Application logging configuration.
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


logger.remove()


# Console
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    colorize=True,
    enqueue=True,
    backtrace=False,
    diagnose=False,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan> | "
        "{message}"
    ),
)


# Application log
logger.add(
    LOG_DIR / "application.log",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
    enqueue=True,
)


# Error log
logger.add(
    LOG_DIR / "error.log",
    level="ERROR",
    rotation="5 MB",
    retention="60 days",
    compression="zip",
    encoding="utf-8",
    enqueue=True,
)