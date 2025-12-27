import argparse
import json
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from src.llm_connector.huggingface_llm import HuggingFaceLLM
from src.utils.logging_utils import setup_logger
from src.utils.schema_loader import load_schema_class
from src.data_processing.entity_matching_processor import EntityMatchingProcessor

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
    parser.add_argument("--count", type=int, help="Number of pairs to procesas (default: all from start-index).")
    return parser.parse_args()

# TODO: Add a file in output with prompt, task_config, hf-model, dataset_config used for reproducibility


def load_dataset_config(dataset_config_path, dataset_name, logger):
    """Load dataset configuration from YAML file."""
    try:
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        
        if dataset_name not in dataset_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in {dataset_config_path}")
        
        dataset = dataset_config[dataset_name]
        logger.info(f"Loaded dataset config for: {dataset_name}")
        
        # Validate required fields
        required_fields = ['d1', 'd2', 'pairs']
        missing_fields = [field for field in required_fields if field not in dataset]
        if missing_fields:
            raise ValueError(f"Missing required fields in dataset config: {missing_fields}")
        
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset config: {e}")
        raise


def save_run_metadata(save_path, args, task_config, dataset_config, prompt_template, logger, 
                      start_time=None, finish_time=None, duration_minutes=None):
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
    
    # Save as JSON
    metadata_file = save_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metadata saved to: {metadata_file}")
    
    # Also save as YAML for easier reading
    metadata_yaml = save_path / "metadata.yaml"
    with open(metadata_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Metadata (YAML) saved to: {metadata_yaml}")
    
    return metadata_file


def _replace_prompt_entities(prompt_template, entity_values):  
    """Helper to replace entity placeholders in prompt template.
    
    Supports both [entity] and {{entity}} format.
    """
    prompt = prompt_template
    for entity, value in entity_values.items():
        prompt = prompt.replace(f"[{entity}]", value)
        prompt = prompt.replace(f"{{{{{entity}}}}}", value)
    return prompt


def _save_partial_results(results, partial_save_path, pairs_processed, total_count, logger):
    """Helper to save partial results to CSV."""
    partial_df = pd.DataFrame(results)
    partial_file = partial_save_path / "partial_results.csv"
    partial_df.to_csv(partial_file, index=False)
    logger.info(f"Partial save completed: {pairs_processed} pairs saved to {partial_file}")
    print(f"Partial save: {pairs_processed}/{total_count} pairs processed")


def _should_partial_save(pairs_processed, partial_save_interval, partial_save_path):
    """Check if it's time to do a partial save."""
    return (partial_save_interval and partial_save_path and 
            pairs_processed % partial_save_interval == 0)


def process_entity_pairs_single(llm, processor, entities_list, prompt_template, schema_class, logger, 
                                start_index=0, count=None, partial_save_interval=None, partial_save_path=None):
    """Process entity pairs one at a time (original behavior)."""
    results = []
    
    total_pairs = len(processor.pairs_df)
    if count is None:
        count = total_pairs - start_index
    
    end_index = min(start_index + count, total_pairs)
    
    # Get dynamic column names from processor
    col1_name = processor.col1_name
    col2_name = processor.col2_name
    
    for pair_idx in range(start_index, end_index):
        logger.info(f"Processing pair {pair_idx + 1}/{total_pairs} (batch: {pair_idx - start_index + 1}/{count})")
        
        try:
            # Get the pair data
            pair_data = processor.get_pair_by_index(pair_idx)
            
            # Build entity values from raw entities (remove 'id' field)
            entity_values = {}
            if len(entities_list) >= 2:
                entity1_clean = {k: v for k, v in pair_data['entity1_raw'].items() if k != 'id'}
                entity2_clean = {k: v for k, v in pair_data['entity2_raw'].items() if k != 'id'}
                
                entity_values[entities_list[0]] = str(entity1_clean)
                entity_values[entities_list[1]] = str(entity2_clean)
            
            # Replace placeholders in prompt
            prompt = _replace_prompt_entities(prompt_template, entity_values)
            
            logger.debug(f"Generated prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            
            response = llm.generate_structured(prompt, schema_class)

            # Create result row with dynamic column names
            result = {
                "row_id": pair_idx + 1,
                "pair_index": pair_data['pair_index'],
                f"{col1_name}_index": pair_data[f'{col1_name}_index'],
                f"{col2_name}_index": pair_data[f'{col2_name}_index'],
                "entity1_raw": str(pair_data['entity1_raw']),
                "entity2_raw": str(pair_data['entity2_raw']),
                "prompt": prompt,
                "response": str(response),
                "success": True
            }
            
            # Add individual fields from the response
            if hasattr(response, 'model_dump'):
                result.update(response.model_dump())
            
            results.append(result)
            logger.info(f"Pair {pair_idx + 1} processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process pair {pair_idx + 1}: {e}")

            # Try to get pair data for error logging
            try:
                pair_data = processor.get_pair_by_index(pair_idx)
                result = {
                    "row_id": pair_idx + 1,
                    "pair_index": pair_data['pair_index'],
                    f"{col1_name}_index": pair_data[f'{col1_name}_index'],
                    f"{col2_name}_index": pair_data[f'{col2_name}_index'],
                    "entity1_raw": str(pair_data['entity1_raw']),
                    "entity2_raw": str(pair_data['entity2_raw']),
                    "prompt": prompt if 'prompt' in locals() else "Error generating prompt",
                    "error": str(e),
                    "success": False
                }
            except:
                result = {
                    "row_id": pair_idx + 1,
                    "pair_index": pair_idx,
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
        
        pairs_processed = pair_idx - start_index + 1
        if _should_partial_save(pairs_processed, partial_save_interval, partial_save_path):
            _save_partial_results(results, partial_save_path, pairs_processed, count, logger)
    
    return pd.DataFrame(results)


def process_entity_pairs_multi(llm, processor, entities_list, prompt_template, schema_class, pairs_per_prompt, logger, 
                               start_index=0, count=None, partial_save_interval=None, partial_save_path=None):
    """Process multiple entity pairs in a single prompt using static schema."""
    results = []
    
    total_pairs = len(processor.pairs_df)
    if count is None:
        count = total_pairs - start_index
    
    end_index = min(start_index + count, total_pairs)
    
    col1_name = processor.col1_name
    col2_name = processor.col2_name
    
    # Process in batches
    batch_num = 0
    for batch_start in range(start_index, end_index, pairs_per_prompt):
        batch_end = min(batch_start + pairs_per_prompt, end_index)
        actual_pairs_in_batch = batch_end - batch_start
        batch_num += 1
        
        logger.info(f"Processing batch {batch_num} with pairs {batch_start + 1}-{batch_end} ({actual_pairs_in_batch} pairs)")
        
        try:
            # Collect all pairs in this batch
            batch_pairs = []
            entity_values = {}
            
            for i, pair_idx in enumerate(range(batch_start, batch_end)):
                pair_data = processor.get_pair_by_index(pair_idx)
                batch_pairs.append(pair_data)
                
                # Create entity placeholders for this pair
                entity1_clean = {k: v for k, v in pair_data['entity1_raw'].items() if k != 'id'}
                entity2_clean = {k: v for k, v in pair_data['entity2_raw'].items() if k != 'id'}
                
                # Use indexed entity names: entity_1a, entity_1b for pair 1, entity_2a, entity_2b for pair 2, etc.
                pair_num = i + 1
                entity_values[f"entity_{pair_num}a"] = str(entity1_clean)
                entity_values[f"entity_{pair_num}b"] = str(entity2_clean)
            
            # Replace placeholders in prompt
            prompt = _replace_prompt_entities(prompt_template, entity_values)
            
            logger.debug(f"Generated prompt for batch: {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
            
            # Get LLM response using the provided schema
            response = llm.generate_structured(prompt, schema_class)
            
            # Parse response and create individual results
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else {}
            
            for i, pair_data in enumerate(batch_pairs):
                pair_num = i + 1
                match_field = f"pair_{pair_num}_match"
                match_value = response_dict.get(match_field, None)
                
                result = {
                    "row_id": pair_data['pair_index'] + 1,
                    "pair_index": pair_data['pair_index'],
                    f"{col1_name}_index": pair_data[f'{col1_name}_index'],
                    f"{col2_name}_index": pair_data[f'{col2_name}_index'],
                    "entity1_raw": str(pair_data['entity1_raw']),
                    "entity2_raw": str(pair_data['entity2_raw']),
                    "batch_num": batch_num,
                    "pairs_in_batch": actual_pairs_in_batch,
                    "prompt": prompt,
                    "response": str(response),
                    "match": match_value,
                    "success": True
                }
                
                results.append(result)
                logger.info(f"Pair {pair_data['pair_index'] + 1} processed successfully (match: {match_value})")
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num} (pairs {batch_start + 1}-{batch_end}): {e}")
            
            # Create error results for all pairs in this batch
            for pair_idx in range(batch_start, batch_end):
                try:
                    pair_data = processor.get_pair_by_index(pair_idx)
                    result = {
                        "row_id": pair_data['pair_index'] + 1,
                        "pair_index": pair_data['pair_index'],
                        f"{col1_name}_index": pair_data[f'{col1_name}_index'],
                        f"{col2_name}_index": pair_data[f'{col2_name}_index'],
                        "entity1_raw": str(pair_data['entity1_raw']),
                        "entity2_raw": str(pair_data['entity2_raw']),
                        "batch_num": batch_num,
                        "pairs_in_batch": actual_pairs_in_batch,
                        "prompt": prompt if 'prompt' in locals() else "Error generating prompt",
                        "error": str(e),
                        "success": False
                    }
                except:
                    result = {
                        "row_id": pair_idx + 1,
                        "pair_index": pair_idx,
                        "batch_num": batch_num,
                        "error": str(e),
                        "success": False
                    }
                
                results.append(result)
        
        # Check for partial save
        pairs_processed = batch_end - start_index
        if _should_partial_save(pairs_processed, partial_save_interval, partial_save_path):
            _save_partial_results(results, partial_save_path, pairs_processed, count, logger)
    
    return pd.DataFrame(results)


def main():
    args = parse_args()
    
    log_path = "./logs.log"
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.mkdir(parents=True, exist_ok=True)
    
    log_to_console = args.log_console or not args.log_file
    logger = setup_logger(to_console=log_to_console, log_file=log_path)
    
    start_time = datetime.now()
    logger.info(f"Run started at: {start_time.isoformat()}")

    # Load dataset config
    dataset = load_dataset_config(DATASET_CONFIG_PATH, args.dataset, logger)
    logger.info(f"Dataset files:")
    logger.info(f"  d1: {dataset['d1']}")
    logger.info(f"  d2: {dataset['d2']}")
    logger.info(f"  pairs: {dataset['pairs']}")
    if 'gt' in dataset:
        logger.info(f"  ground truth: {dataset['gt']}")
    
    # Load task config
    with open(TASK_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    task_config = config.get(args.task)
    if not task_config:
        raise ValueError(f"Task '{args.task}' not found in {TASK_CONFIG_PATH}")
    
    logger.info(f"Task config: {task_config}")
    
    # Extract task configuration
    schema_path = task_config["schema"]
    entities_list = task_config["entities"]
    prompt_path = task_config["prompt"]
    temperature = task_config.get("temperature", 0.5)
    max_tokens = task_config.get("max_tokens", 256)
    
    # Load schema class
    try:
        schema_class = load_schema_class(schema_path)
        logger.info(f"Loaded schema: {schema_path}")
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        sys.exit(1)
    
    # Initialize entity matching processor
    try:
        processor = EntityMatchingProcessor(dataset['d1'], dataset['d2'], dataset['pairs'])
        logger.info("Entity matching processor initialized successfully")
        logger.info(f"Loaded {len(processor.pairs_df)} pairs for processing")
        logger.info(f"Dataset column names: {processor.col1_name}, {processor.col2_name}")
    except Exception as e:
        logger.error(f"Failed to initialize entity matching processor: {e}")
        sys.exit(1)
    
    # Load prompt template
    prompt_template = Path(prompt_path).read_text(encoding='utf-8')
    logger.info(f"Loaded prompt template from: {prompt_path}")
    
    # Determine processing mode
    pairs_per_prompt = task_config.get("pairs_per_prompt",None)
    if pairs_per_prompt > 1:
        logger.info(f"Multi-pair mode: processing {pairs_per_prompt} pairs per prompt")
    else:
        logger.info("Single-pair mode: processing 1 pair per prompt")
    
    # Initialize model
    logger.info(f"Loading Hugging Face model: {args.hf_model}")
    llm = HuggingFaceLLM(
        model_name=args.hf_model,
        temperature=temperature,
        max_tokens=max_tokens,
        hf_token=HF_TOKEN
    )
    logger.info("Model ready.")
    
    # Setup partial save path
    partial_save_path = None
    if args.partial_save and args.save:
        partial_save_path = Path(args.save)
        partial_save_path.mkdir(parents=True, exist_ok=True)

        # Save initial metadata (without finish time)
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
    
    # Process entity pairs
    if pairs_per_prompt > 1:
        results_df = process_entity_pairs_multi(
            llm, processor, entities_list, prompt_template, schema_class, pairs_per_prompt, logger,
            start_index=args.start_index,
            count=args.count,
            partial_save_interval=args.partial_save if args.save else None,
            partial_save_path=partial_save_path
        )
    else:
        results_df = process_entity_pairs_single(
            llm, processor, entities_list, prompt_template, schema_class, logger,
            start_index=args.start_index,
            count=args.count,
            partial_save_interval=args.partial_save if args.save else None,
            partial_save_path=partial_save_path
        )
    
    # Print summary
    successful = results_df['success'].sum()
    total = len(results_df)
    logger.info(f"Processing completed: {successful}/{total} successful")
    print(f"\nProcessing Summary:")
    print(f"Total pairs: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    # After processing completes, record finish time
    finish_time = datetime.now()
    duration = (finish_time - start_time).total_seconds() / 60
    logger.info(f"Run finished at: {finish_time.isoformat()}")
    logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save final metadata with timing info
        save_run_metadata(
            save_path=save_path,
            args=args,
            task_config=task_config,
            dataset_config=dataset,
            prompt_template=prompt_template,
            logger=logger,
            start_time=start_time,
            finish_time=finish_time,
            duration_minutes=duration
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
    
    # Display sample results
    if len(results_df) > 0:
        print(f"\nFirst few results:")
        display_columns = ['pair_index', 'match', 'success']
        if pairs_per_prompt > 1:
            display_columns.insert(1, 'batch_num')
        
        available_columns = [col for col in display_columns if col in results_df.columns]
        
        if available_columns:
            print(results_df[available_columns].head(10).to_string())
        else:
            print(results_df.head(10))
        
        if 'match' in results_df.columns:
            match_summary = results_df['match'].value_counts()
            print(f"\nMatch Summary:")
            print(f"Matches (1): {match_summary.get(1, 0)}")
            print(f"No matches (0): {match_summary.get(0, 0)}")
        
        print(f"\nAll available columns: {list(results_df.columns)}")

if __name__ == "__main__":
    main()