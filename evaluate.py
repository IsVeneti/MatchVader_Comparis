import argparse
import os
from llama_cpp import Path
import yaml
from src.data_processing.pairs_to_ids import get_or_create_id_pairs
from src.evaluator.data_manipulation import load_ground_truth, merge_dataframes, merge_response_pairs, merge_response_pairs_partial
from src.evaluator.evaluate import evaluate_results, save_results_as_json
import pandas as pd

# Global config variable
DS_CONFIG = "configs/dataset_config.yaml"

# Argument parser setup
parser = argparse.ArgumentParser(description='Evaluate candidate pairs against ground truth')
parser.add_argument('--dataset', '-d',choices=[f'dataset_{i}' for i in range(1, 10)], required=True, help='Which dataset to use from config')
parser.add_argument('--response_csv', '-r', help='LLM response CSV file',  required=True)
parser.add_argument('--join', '-j', choices=['inner', 'outer', 'left', 'right'], default='inner', help='Join type for comparison (default: inner)')
parser.add_argument('--partial', '-p', action='store_true', 
                   help='Handle partial runs (merge by pair_index instead of position)')

def run_evaluation(pairs_with_ids, gt_csv, join_type='inner'):
    """Run complete evaluation pipeline."""
    matches = merge_dataframes(pairs_with_ids, gt_csv, join_type=join_type)
    metrics = evaluate_results(matches)
    return metrics


if __name__ == '__main__':
    # Load config
    with open(DS_CONFIG, 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    args = parser.parse_args()
    
    # Get paths from config
    dataset_config = CONFIG[args.dataset]
    pairs_path = dataset_config['pairs_with_ids']
    print(pairs_path)
    ground_truth_path = dataset_config['gt']
    if not os.path.exists(pairs_path):
        print("here?")
        pairs_with_ids = get_or_create_id_pairs(dataset_config['pairs'],
                               dataset_config['d1'],
                               dataset_config['d2'],
                               output_csv=pairs_path,
                               force_recreate=False)
        pairs_with_ids.to_csv(pairs_path, index=False)

    
    # pairs_from_response = filter_csv_columns(args.response_csv, columns=['id1', 'id2', 'match'])

    # Process and evaluate
    if args.partial:
        pairs_from_response = merge_response_pairs_partial(args.response_csv, pairs_path)
    else:
        pairs_from_response = merge_response_pairs(args.response_csv, pairs_path)
    ground_truth_df = load_ground_truth(ground_truth_path)
    
    metrics = run_evaluation(pairs_from_response, ground_truth_df, join_type=args.join)
    
    print(metrics)
    
    # Save output in same folder as response CSV
    response_path = Path(args.response_csv)
    output_path = response_path.parent / f"{response_path.stem}_{args.join}_eval.json"
    save_results_as_json(metrics, str(output_path))