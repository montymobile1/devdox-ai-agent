import os
import logging
from logging.handlers import RotatingFileHandler
from app.config import settings

def setup_logging():
    """
    Configures logging for the entire application.
    Logs messages to console and to a rotating log file.
    """
    # Create a custom logger
    logger = logging.getLogger()
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    logger.setLevel(log_level)

    # Prevent duplicate logs during development when using `uvicorn --reload`.
    # Each reload reinitializes the logger and adds new handlers unless cleared first.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt=("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )

    # Console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    log_dir = settings.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    # File handler with rotating log (max 5MB per file, keeping 3 backup files)
    file_handler = RotatingFileHandler(
         os.path.join(log_dir, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optionally, create more handlers (e.g., for error logs, different log levels, etc.)
    error_handler = logging.FileHandler(  os.path.join(log_dir, "error.log"))
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger
