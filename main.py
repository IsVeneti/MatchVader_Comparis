import argparse
import sys
from pathlib import Path
from pydantic import BaseModel

from src.llm_connector.huggingface_llm import HuggingFaceLLM
from src.utils.logging_utils import setup_logger
from src.utils.config_loader import load_task_config
from src.utils.schema_loader import load_schema_class


def parse_args():
    parser = argparse.ArgumentParser(description="Run HuggingFaceLLM with structured prompt output.")
    parser.add_argument("--hf-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Hugging Face model name or path.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., Pairs, Triples) from the config file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--log-file", type=str, help="Path to log file.")
    parser.add_argument("--log-console", action="store_true", help="Enable console logging (default if no log file).")
    return parser.parse_args()


def run_prompt_loop(model: HuggingFaceLLM, schema_class: type[BaseModel], logger):
    logger.info("Structured output mode is enabled.")
    print("\nEnter a prompt (Ctrl+C to exit):")

    try:
        while True:
            prompt = input("\nPrompt > ")
            logger.info(f"Prompt received: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            try:
                response = model.generate_structured(prompt, schema_class)
                print("Structured Output:")
                print(response)
                logger.info(f"Parsed output: {response}")
            except Exception as e:
                logger.error(f"Failed to generate or parse structured output: {e}")
    except KeyboardInterrupt:
        logger.info("Exiting.")
        print()


def main():
    args = parse_args()

    # Logging
    log_to_console = args.log_console or not args.log_file
    logger = setup_logger(to_console=log_to_console, log_file=args.log_file)

    # Load task config
    task_config = load_task_config(Path(args.config), args.task, logger)
    schema_path = task_config.get("schema")
    temperature = task_config.get("temperature", 0.0)
    max_tokens = task_config.get("max_tokens", 256)

    # Load schema
    try:
        schema_class = load_schema_class(schema_path)
        logger.info(f"Loaded schema: {schema_path}")
    except Exception as e:
        logger.exception(f"Failed to load schema: {e}")
        sys.exit(1)

    # Load model
    logger.info(f"Loading Hugging Face model: {args.hf_model}")
    model = HuggingFaceLLM(
        model_name=args.hf_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logger.info("Model ready.")

    # Run interaction
    run_prompt_loop(model, schema_class, logger)


if __name__ == "__main__":
    main()
