import argparse
import sys
import yaml
from pathlib import Path

from dotenv import load_dotenv
import os

from src.llm_connector.huggingface_llm import HuggingFaceLLM
from src.utils.logging_utils import setup_logger
from src.utils.schema_loader import load_schema_class
from src.utils.run_config import load_dataset_config, save_run_metadata
from src.data_processing.entity_matching_processor import EntityMatchingProcessor
from src.processing.pair_processor import process_entity_pairs_single, process_entity_pairs_multi
from src.processing.candidate_processor import process_candidate_selection

from datetime import datetime

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
DATASET_CONFIG_PATH = "configs/dataset_config.yaml"
TASK_CONFIG_PATH = "configs/task_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run HuggingFaceLLM with structured prompt output for entity matching.")
    parser.add_argument("--hf-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Hugging Face model name or path.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., Pairs) from the task config file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., dataset_1, dataset_2) from the dataset config file.")
    parser.add_argument("--log-file", type=str, help="Path to log file.")
    parser.add_argument("--log-console", action="store_true", help="Enable console logging (default if no log file).")
    parser.add_argument("--save", type=str, help="Path to save results (optional).")
    parser.add_argument("--partial-save", type=int, help="Save results every X entries (enables partial saving).")
    parser.add_argument("--start-index", type=int, default=0, help="Starting pair index for processing.")
    parser.add_argument("--count", type=int, help="Number of pairs to process (default: all from start-index).")
    parser.add_argument("--repetitions", type=int, default=0,
                        help="Number of times to repeat the instruction within the prompt (0 = no repetition, 1 = repeat once, etc.).")
    parser.add_argument("--repetition-style", type=str, default="basic", choices=["basic", "verbose"],
                        help="'basic' = space-separated copies, 'verbose' = bridging phrases between copies.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_file or args.log_console:
        log_to_console = args.log_console
        log_file_path = Path(args.log_file) if args.log_file else None
        if log_file_path:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_to_console = True
        log_file_path = Path("./logs.log")
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(to_console=log_to_console, log_file=log_file_path)

    start_time = datetime.now()
    logger.info(f"Run started at: {start_time.isoformat()}")

    dataset = load_dataset_config(DATASET_CONFIG_PATH, args.dataset, logger)
    logger.info(f"Dataset files:")
    logger.info(f"  d1: {dataset['d1']}")
    logger.info(f"  d2: {dataset['d2']}")
    logger.info(f"  pairs: {dataset['pairs']}")
    if 'gt' in dataset:
        logger.info(f"  ground truth: {dataset['gt']}")

    with open(TASK_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    task_config = config.get(args.task)
    if not task_config:
        raise ValueError(f"Task '{args.task}' not found in {TASK_CONFIG_PATH}")

    logger.info(f"Task config: {task_config}")

    schema_path = task_config["schema"]
    entities_list = task_config["entities"]
    prompt_path = task_config["prompt"]
    temperature = task_config.get("temperature", 0.5)
    max_tokens = task_config.get("max_tokens", 256)

    try:
        schema_class = load_schema_class(schema_path)
        logger.info(f"Loaded schema: {schema_path}")
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        sys.exit(1)

    try:
        processor = EntityMatchingProcessor(dataset['d1'], dataset['d2'], dataset['pairs_with_ids'])
        logger.info("Entity matching processor initialized successfully")
        logger.info(f"Loaded {len(processor.pairs_df)} pairs for processing")
        logger.info(f"Dataset column names: {processor.col1_name}, {processor.col2_name}")
    except Exception as e:
        logger.error(f"Failed to initialize entity matching processor: {e}")
        sys.exit(1)

    prompt_template = Path(prompt_path).read_text(encoding='utf-8')
    logger.info(f"Loaded prompt template from: {prompt_path}")

    task_type = task_config.get("task_type", "pairs")
    pairs_per_prompt = task_config.get("pairs_per_prompt", None)

    logger.info(f"Loading Hugging Face model: {args.hf_model}")
    llm = HuggingFaceLLM(
        model_name=args.hf_model,
        temperature=temperature,
        max_tokens=max_tokens,
        hf_token=HF_TOKEN
    )
    logger.info("Model ready.")

    partial_save_path = None
    if args.partial_save and args.save:
        partial_save_path = Path(args.save)
        partial_save_path.mkdir(parents=True, exist_ok=True)
        save_run_metadata(
            save_path=partial_save_path,
            args=args,
            task_config=task_config,
            dataset_config=dataset,
            prompt_template=prompt_template,
            logger=logger,
            start_time=start_time
        )
        logger.info(f"Partial save enabled: saving every {args.partial_save} pairs to {partial_save_path}")
    elif args.partial_save and not args.save:
        logger.warning("--partial-save requires --save to be set. Partial saving disabled.")

    if task_type == "candidate_selection":
        logger.info(f"Running candidate selection task (target_side: {task_config.get('target_side', 'd2')})")
        results_df = process_candidate_selection(
            llm, processor, prompt_template, schema_class, task_config, logger,
            start_index=args.start_index,
            count=args.count,
            partial_save_interval=args.partial_save if args.save else None,
            partial_save_path=partial_save_path,
            repetitions=args.repetitions,
            repetition_style=args.repetition_style
        )
    else:
        pairs_per_prompt = task_config.get("pairs_per_prompt", 1)
        if pairs_per_prompt > 1:
            logger.info(f"Multi-pair mode: processing {pairs_per_prompt} pairs per prompt")
            results_df = process_entity_pairs_multi(
                llm, processor, entities_list, prompt_template, schema_class, pairs_per_prompt, logger,
                start_index=args.start_index, count=args.count,
                partial_save_interval=args.partial_save if args.save else None,
                partial_save_path=partial_save_path,
                repetitions=args.repetitions,
                repetition_style=args.repetition_style
            )
        else:
            logger.info("Single-pair mode: processing 1 pair per prompt")
            results_df = process_entity_pairs_single(
                llm, processor, entities_list, prompt_template, schema_class, logger,
                start_index=args.start_index, count=args.count,
                partial_save_interval=args.partial_save if args.save else None,
                partial_save_path=partial_save_path,
                repetitions=args.repetitions,
                repetition_style=args.repetition_style
            )

    successful = results_df['success'].sum()
    total = len(results_df)
    logger.info(f"Processing completed: {successful}/{total} successful")
    print(f"\nProcessing Summary:")
    print(f"Total pairs: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")

    finish_time = datetime.now()
    duration = (finish_time - start_time).total_seconds() / 60
    logger.info(f"Run finished at: {finish_time.isoformat()}")
    logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    token_stats = llm.get_token_stats()
    logger.info(f"Token usage: {token_stats}")

    if args.save:
        save_path = Path(args.save)
        save_path.mkdir(parents=True, exist_ok=True)

        save_run_metadata(
            save_path=save_path,
            args=args,
            task_config=task_config,
            dataset_config=dataset,
            prompt_template=prompt_template,
            logger=logger,
            start_time=start_time,
            finish_time=finish_time,
            duration_minutes=duration,
            token_stats=token_stats
        )

        results_file = save_path / "results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to: {results_file}")
        print(f"Results saved to: {results_file}")

        successful_results = results_df[results_df['success'] == True]
        if len(successful_results) > 0:
            successful_file = save_path / "successful_results.csv"
            successful_results.to_csv(successful_file, index=False)
            logger.info(f"Successful results saved to: {successful_file}")

        partial_file = save_path / "partial_results.csv"
        if partial_file.exists():
            try:
                partial_file.unlink()
                logger.info(f"Deleted partial save file: {partial_file}")
                print(f"Cleaned up partial save file")
            except Exception as e:
                logger.warning(f"Could not delete partial save file: {e}")

    if len(results_df) > 0:
        print(f"\nFirst few results:")
        display_columns = ['pair_index', 'match', 'success']
        if pairs_per_prompt and pairs_per_prompt > 1:
            display_columns.insert(1, 'batch_num')

        available_columns = [col for col in display_columns if col in results_df.columns]

        if available_columns:
            print(results_df[available_columns].head(10).to_string())
        else:
            print(results_df.head(10))

        if 'match' in results_df.columns:
            match_summary = results_df['match'].value_counts().to_dict()
            print(f"\nMatch Summary:")
            print(f"Matches (1): {match_summary.get(1, 0)}")
            print(f"No matches (0): {match_summary.get(0, 0)}")

        print(f"\nAll available columns: {list(results_df.columns)}")


if __name__ == "__main__":
    main()
