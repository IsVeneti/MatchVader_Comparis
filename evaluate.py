import argparse
from llama_cpp import Path
import yaml
from src.evaluator.data_manipulation import merge_dataframes, merge_response_pairs
from src.evaluator.evaluate import evaluate_results, save_results_as_json
import pandas as pd

# Global config variable
DS_CONFIG = "configs/dataset_config.yaml"

# Argument parser setup
parser = argparse.ArgumentParser(description='Evaluate candidate pairs against ground truth')
parser.add_argument('--dataset', '-d',choices=['dataset_1', 'dataset_2'], required=True, help='Which dataset to use from config')
parser.add_argument('--response_csv', '-r', help='LLM response CSV file',  required=True)
parser.add_argument('--join', '-j', choices=['inner', 'outer', 'left', 'right'], default='inner', help='Join type for comparison (default: inner)')


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
    pairs_path = dataset_config['pairs']
    ground_truth_path = dataset_config['gt']
    
    # Process and evaluate
    pairs_from_response = merge_response_pairs(args.response_csv, pairs_path)
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    metrics = run_evaluation(pairs_from_response, ground_truth_df, join_type=args.join)
    
    print(metrics)
    
    # Save output in same folder as response CSV
    response_path = Path(args.response_csv)
    output_path = response_path.parent / f"{response_path.stem}_{args.join}_eval.json"
    save_results_as_json(metrics, str(output_path))
    print(f"\n✓ Saved results to '{output_path}'")