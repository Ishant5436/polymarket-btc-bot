"""
Structured logging configuration.
Uses structlog for JSON-formatted, production-ready logging.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = None, json_output: bool = False):
    """
    Configure logging for the application.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Default: INFO.
        json_output: If True, output logs as JSON (for production). 
                     If False, use human-readable format (for development).
    """
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)

    if json_output:
        # JSON format for production / journalctl
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"module":"%(name)s","message":"%(message)s"}'
        )
    else:
        # Human-readable for development
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    console.setFormatter(formatter)
    root.addHandler(console)

    # Suppress noisy third-party loggers
    for noisy in ("websockets", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured | level=%s json=%s", level or "INFO", json_output
    )
