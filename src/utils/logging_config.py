"""Logging configuration for NutriBot."""

import logging
import logging.handlers
import os
from config.settings import get_config


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    config = get_config()
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.log_level))

    # Create formatters and handlers
    formatter = logging.Formatter(config.log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(config.log_dir, f"{name.split('.')[-1]}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, config.log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
