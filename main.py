import argparse
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from dotenv import load_dotenv
from src.llm_connector.huggingface_llm import HuggingFaceLLM
from src.utils.logging_utils import setup_logger
from src.utils.schema_loader import load_schema_class
from src.data_processing.entity_matching_processor import EntityMatchingProcessor  # Import your processor

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

def parse_args():
    parser = argparse.ArgumentParser(description="Run HuggingFaceLLM with structured prompt output for entity matching.")
    parser.add_argument("--hf-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Hugging Face model name or path.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., Pairs) from the config file.")
    parser.add_argument("--rest1-file", type=str, default="data/rest1clean.csv", help="Path to rest1clean.csv file.")
    parser.add_argument("--rest2-file", type=str, default="data/rest2clean.csv", help="Path to rest2clean.csv file.")
    parser.add_argument("--pairs-file", type=str, default="data/d1_pairs.csv", help="Path to d1_pairs.csv file.")
    parser.add_argument("--config", type=str, default="task_config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--log-file", type=str, help="Path to log file.")
    parser.add_argument("--log-console", action="store_true", help="Enable console logging (default if no log file).")
    parser.add_argument("--save", type=str, help="Path to save results (optional).")
    parser.add_argument("--start-index", type=int, default=0, help="Starting pair index for processing.")
    parser.add_argument("--count", type=int, help="Number of pairs to process (default: all from start-index).")
    return parser.parse_args()



def process_entity_pairs(llm, processor, entities_list, prompt_template, schema_class, logger, start_index=0, count=None):
    """Process entity pairs and generate structured outputs."""
    results = []
    
    total_pairs = len(processor.pairs_df)
    if count is None:
        count = total_pairs - start_index
    
    end_index = min(start_index + count, total_pairs)
    
    for pair_idx in range(start_index, end_index):
        logger.info(f"Processing pair {pair_idx + 1}/{total_pairs} (batch: {pair_idx - start_index + 1}/{count})")
        
        try:
            # Get the pair data
            pair_data = processor.get_pair_by_index(pair_idx)
            
            # Replace entities in prompt template based on entities_list
            prompt = prompt_template
            entity_values = {}
            
            # Map the entities from config to the formatted entities
            if len(entities_list) >= 2:
                entity1_clean = {k: v for k, v in pair_data['entity1_raw'].items() if k != 'id'}
                entity2_clean = {k: v for k, v in pair_data['entity2_raw'].items() if k != 'id'}
                
                entity_values[entities_list[0]] = str(entity1_clean)
                entity_values[entities_list[1]] = str(entity2_clean)
                
                # Replace placeholders in prompt
                for entity in entities_list:
                    if entity in entity_values:
                        # Support both single and double bracket formats
                        prompt = prompt.replace(f"[{entity}]", entity_values[entity])
                        prompt = prompt.replace(f"{{{{{entity}}}}}", entity_values[entity])
            
            logger.debug(f"Generated prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            
            response = llm.generate_structured(prompt, schema_class)
            
            # Create result row
            result = {
                "row_id": pair_idx + 1,
                "pair_index": pair_data['pair_index'],
                "rest1_index": pair_data['rest1_index'],
                "rest2_index": pair_data['rest2_index']
            }
            
            # Add entity values
            result.update(entity_values)
            
            result.update({
                "entity1_formatted": pair_data['entity1_formatted'],
                "entity2_formatted": pair_data['entity2_formatted'],
                "entity1_raw": str(pair_data['entity1_raw']),
                "entity2_raw": str(pair_data['entity2_raw']),
                "prompt": prompt,
                "response": str(response),
                "success": True
            })
            
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
                    "rest1_index": pair_data['rest1_index'],
                    "rest2_index": pair_data['rest2_index'],
                    "entity1_formatted": pair_data['entity1_formatted'],
                    "entity2_formatted": pair_data['entity2_formatted'],
                    "prompt": prompt if 'prompt' in locals() else "Error generating prompt",
                    "error": str(e),
                    "success": False
                }
                if len(entities_list) >= 2:
                    result[entities_list[0]] = pair_data['entity1_formatted']
                    result[entities_list[1]] = pair_data['entity2_formatted']
            except:
                result = {
                    "row_id": pair_idx + 1,
                    "pair_index": pair_idx,
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
    
    return pd.DataFrame(results)

def main():
    args = parse_args()
    log_path = "./logs.log"
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_to_console = args.log_console or not args.log_file
    logger = setup_logger(to_console=log_to_console, log_file=log_path)
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    task_config = config.get(args.task)
    if not task_config:
        raise ValueError(f"Task '{args.task}' not found in {args.config}")
    
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
    
    # Load prompt template
    prompt_template = Path(prompt_path).read_text(encoding='utf-8')
    logger.info(f"Loaded prompt template from: {prompt_path}")
    
    # Initialize entity matching processor
    try:
        processor = EntityMatchingProcessor(args.rest1_file, args.rest2_file, args.pairs_file)
        logger.info("Entity matching processor initialized successfully")
        logger.info(f"Loaded {len(processor.pairs_df)} pairs for processing")
    except Exception as e:
        logger.error(f"Failed to initialize entity matching processor: {e}")
        sys.exit(1)
    
    # Validate entities list for entity matching
    expected_entities = ["entity_1", "entity_2"]
    if not all(entity in entities_list for entity in expected_entities):
        logger.warning(f"Expected entities {expected_entities}, got {entities_list}")
        logger.info("Proceeding with provided entities list")
    
    print(f"HF_TOKEN {HF_TOKEN}")
    # Initialize model
    logger.info(f"Loading Hugging Face model: {args.hf_model}")
    llm = HuggingFaceLLM(
        model_name=args.hf_model,
        temperature=temperature,
        max_tokens=max_tokens,
        hf_token=HF_TOKEN

    )
    logger.info("Model ready.")
    
    # Process entity pairs
    results_df = process_entity_pairs(
        llm, processor, entities_list, prompt_template, schema_class, logger,
        start_index=args.start_index,
        count=args.count
    )
    
    # Print summary
    successful = results_df['success'].sum()
    total = len(results_df)
    logger.info(f"Processing completed: {successful}/{total} successful")
    print(f"\nProcessing Summary:")
    print(f"Total pairs: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    # Save results if requested
    if args.save:
        save_path = Path(args.save)
        save_path.mkdir(parents=True, exist_ok=True)
        
        results_file = save_path / "results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to: {results_file}")
        print(f"Results saved to: {results_file}")
        
        # Save successful results only
        successful_results = results_df[results_df['success'] == True]
        if len(successful_results) > 0:
            successful_file = save_path / "successful_results.csv"
            successful_results.to_csv(successful_file, index=False)
            logger.info(f"Successful results saved to: {successful_file}")
    
    # Display sample results
    if len(results_df) > 0:
        print(f"\nFirst few results:")
        # Show the most important columns including LLM responses
        display_columns = ['pair_index', 'entity_1', 'entity_2', 'match', 'llm_response_raw', 'success']
        # Use available columns if the expected ones don't exist
        available_columns = [col for col in display_columns if col in results_df.columns]
        
        if 'match' not in available_columns and 'match' in results_df.columns:
            available_columns.append('match')
        
        if not available_columns:
            available_columns = ['pair_index', 'entity1_formatted', 'entity2_formatted', 'success']
            available_columns = [col for col in available_columns if col in results_df.columns]
        
        if available_columns:
            print(results_df[available_columns].head().to_string())
        else:
            print(results_df.head())
        
        # Also show a summary of LLM responses
        if 'match' in results_df.columns:
            match_summary = results_df['match'].value_counts()
            print(f"\nLLM Response Summary:")
            print(f"Matches (1): {match_summary.get(1, 0)}")
            print(f"No matches (0): {match_summary.get(0, 0)}")
        
        # Show columns available in results
        print(f"\nAll available columns in results: {list(results_df.columns)}")

if __name__ == "__main__":
    main()