import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

# Log format constants
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO


def get_logger(log_dir: str, name: str, log_filename: str = None, level: int = LOG_LEVEL) -> logging.Logger:
    """Create and configure a logger.

    Args:
        log_dir: Directory to save log files
        name: Logger name
        log_filename: Log file name (auto-generated if None)
        level: Log level, defaults to INFO

    Returns:
        Configured Logger object
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # If log filename is not specified, auto-generate using timestamp
    if log_filename is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_filename = f"{timestamp}.log"
    else:
        log_filename = f"{log_filename}.log" if not log_filename.endswith('.log') else log_filename

    # Get or create logger
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler (using RotatingFileHandler for log rotation)
    log_path = os.path.join(log_dir, log_filename)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=50,  # Keep 50 backups
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    @contextmanager
    def no_time():
        old_formatters = [h.formatter for h in logger.handlers]
        for h in logger.handlers:
            h.setFormatter(logging.Formatter("%(message)s"))
        try:
            yield
        finally:
            for h, f in zip(logger.handlers, old_formatters):
                h.setFormatter(f)

    logger.no_time = no_time

    with logger.no_time():
        logger.info( "=" * 25 + "   Settings   " + "=" * 25 )
    # Log the log file path
    logger.info(f"Log File Path: {log_path}")

    return logger
