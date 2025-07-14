import sys
import yaml
from pathlib import Path
import logging


def load_task_config(config_path: Path, task_name: str, logger: logging.Logger) -> dict:
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from {config_path}")
    with config_path.open("r") as f:
        full_config = yaml.safe_load(f)

    if task_name not in full_config:
        logger.error(f"Task '{task_name}' not found in config.")
        sys.exit(1)

    return full_config[task_name]
