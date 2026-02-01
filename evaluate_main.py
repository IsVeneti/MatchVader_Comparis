import argparse
import os
from pathlib import Path
import yaml
from src.data_processing.pairs_to_ids import get_or_create_id_pairs
from src.evaluator.data_manipulation import load_ground_truth, merge_dataframes, merge_response_pairs, merge_response_pairs_partial
from src.evaluator.evaluate import evaluate_results, save_results_as_json
import pandas as pd

# Global config variable
DS_CONFIG = "configs/dataset_config.yaml"

def run_evaluation(pairs_with_ids, gt_csv, join_type='inner'):
    """Run complete evaluation pipeline."""
    matches = merge_dataframes(pairs_with_ids, gt_csv, join_type=join_type)
    metrics = evaluate_results(matches)
    return metrics

def evaluate_file(dataset, response_csv, join_type='inner', partial=False, config_path=DS_CONFIG):
    """
    Evaluate a single response CSV file.
    
    Args:
        dataset: Dataset name (e.g., 'dataset_1')
        response_csv: Path to the response CSV file
        join_type: Type of join ('inner', 'outer', 'left', 'right')
        partial: Whether to handle partial runs
        config_path: Path to dataset config YAML
    
    Returns:
        dict: Evaluation metrics
    """
    # Load config
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    # Get paths from config
    dataset_config = CONFIG[dataset]
    pairs_path = dataset_config['pairs_with_ids']
    ground_truth_path = dataset_config['gt']
    
    if not os.path.exists(pairs_path):
        pairs_with_ids = get_or_create_id_pairs(
            dataset_config['pairs'],
            dataset_config['d1'],
            dataset_config['d2'],
            output_csv=pairs_path,
            force_recreate=False
        )
        pairs_with_ids.to_csv(pairs_path, index=False)
    
    # Process and evaluate
    if partial:
        pairs_from_response = merge_response_pairs_partial(response_csv, pairs_path)
    else:
        pairs_from_response = merge_response_pairs(response_csv, pairs_path)
    
    ground_truth_df = load_ground_truth(ground_truth_path)
    metrics = run_evaluation(pairs_from_response, ground_truth_df, join_type=join_type)
    
    # Save output in same folder as response CSV
    response_path = Path(response_csv)
    output_path = response_path.parent / f"{response_path.stem}_{join_type}_eval.json"
    save_results_as_json(metrics, str(output_path))
    
    return metrics

if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Evaluate candidate pairs against ground truth')
    parser.add_argument('--dataset', '-d', choices=[f'dataset_{i}' for i in range(1, 10)], 
                       required=True, help='Which dataset to use from config')
    parser.add_argument('--response_csv', '-r', help='LLM response CSV file', required=True)
    parser.add_argument('--join', '-j', choices=['inner', 'outer', 'left', 'right'], 
                       default='inner', help='Join type for comparison (default: inner)')
    parser.add_argument('--partial', '-p', action='store_true', 
                       help='Handle partial runs (merge by pair_index instead of position)')
    
    args = parser.parse_args()
    
    metrics = evaluate_file(
        dataset=args.dataset,
        response_csv=args.response_csv,
        join_type=args.join,
        partial=args.partial
    )
    
    print(metrics)
