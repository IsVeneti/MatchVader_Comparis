import json
import yaml
from datetime import datetime
from pathlib import Path


def load_dataset_config(dataset_config_path, dataset_name, logger):
    """Load dataset configuration from YAML file."""
    try:
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)

        if dataset_name not in dataset_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in {dataset_config_path}")

        dataset = dataset_config[dataset_name]
        logger.info(f"Loaded dataset config for: {dataset_name}")

        required_fields = ['d1', 'd2', 'pairs']
        missing_fields = [field for field in required_fields if field not in dataset]
        if missing_fields:
            raise ValueError(f"Missing required fields in dataset config: {missing_fields}")

        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset config: {e}")
        raise


def save_run_metadata(save_path, args, task_config, dataset_config, prompt_template, logger,
                      start_time=None, finish_time=None, duration_minutes=None,
                      token_stats=None):
    """
    Save all run configuration for reproducibility.

    Args:
        save_path: Path object where results are saved
        args: Parsed command line arguments
        task_config: Task configuration dict from YAML
        dataset_config: Dataset configuration dict from YAML
        prompt_template: The actual prompt template text
        logger: Logger instance
        start_time: Optional datetime when processing started
        finish_time: Optional datetime when processing finished
        duration_minutes: Optional duration in minutes
    """
    metadata = {
        "run_info": {
            "started_at": start_time.isoformat() if start_time else None,
            "finished_at": finish_time.isoformat() if finish_time else None,
            "duration_minutes": duration_minutes,
            "created_at": datetime.now().isoformat()
        },
        "model": {
            "hf_model": args.hf_model,
            "temperature": task_config.get("temperature", 0.5),
            "max_tokens": task_config.get("max_tokens", 256)
        },
        "token_usage": token_stats if token_stats else {},
        "task": {
            "name": args.task,
            "config": task_config,
            "prompt_template": prompt_template
        },
        "dataset": {
            "name": args.dataset,
            "config": dataset_config
        },
        "processing": {
            "start_index": args.start_index,
            "count": args.count,
            "pairs_per_prompt": task_config.get("pairs_per_prompt", 1)
        },
        "command_line_args": vars(args)
    }

    metadata_file = save_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to: {metadata_file}")

    metadata_yaml = save_path / "metadata.yaml"
    with open(metadata_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Metadata (YAML) saved to: {metadata_yaml}")

    return metadata_file
