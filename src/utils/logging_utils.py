import logging
import sys


def setup_logger(to_console: bool = True, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger("llm_runner")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", 
        "%Y-%m-%d %H:%M:%S"  # Changed from "%H:%M:%S"
    )
    logger.handlers.clear()
    if to_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger