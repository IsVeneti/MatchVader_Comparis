import argparse
import json
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

def _merge_candidate_selection(response_csv, ground_truth_path, join_type):
    """
    ID-based merge for candidate selection results.

    Candidate selection results.csv contains target_id and candidate_id columns
    so we can join directly on those instead of relying on row-order alignment
    with pairs_with_ids (which has a different sort order).
    """
    res_df = pd.read_csv(response_csv, encoding="utf-8")
    if "target_id" not in res_df.columns or "candidate_id" not in res_df.columns:
        raise ValueError(
            f"Candidate selection results must have target_id and candidate_id columns: {response_csv}"
        )

    # Read target_side from the metadata.json saved alongside results.csv.
    metadata_path = Path(response_csv).parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found next to {response_csv}. "
            "Cannot determine target_side for candidate selection evaluation."
        )
    with open(metadata_path) as f:
        meta = json.load(f)
    target_side = meta["task"]["config"]["target_side"]  # "d1" or "d2"
    target_is_d1 = (target_side == "d1")

    # pairs_with_ids: id1=d1-side, id2=d2-side (same as GT D1/D2 order)
    if target_is_d1:
        # D1Target: target=d1->id1, candidate=d2->id2
        pairs_df = res_df[["target_id", "candidate_id", "match"]].rename(
            columns={"target_id": "id1", "candidate_id": "id2"}
        )
    else:
        # D2Target: target=d2->id2, candidate=d1->id1
        pairs_df = res_df[["candidate_id", "target_id", "match"]].rename(
            columns={"candidate_id": "id1", "target_id": "id2"}
        )

    pairs_df["match"] = pd.to_numeric(pairs_df["match"], errors="coerce").fillna(0).astype(int)
    ground_truth_df = load_ground_truth(ground_truth_path)
    return merge_dataframes(pairs_df, ground_truth_df, join_type=join_type)


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

    # Candidate selection results carry target_id/candidate_id so we merge by ID,
    # not by row order (merge_response_pairs assumes row order == pairs_with_ids order,
    # which is false for candidate selection since results are grouped by target).
    res_df_check = pd.read_csv(response_csv, nrows=1)
    is_candidate_selection = "target_id" in res_df_check.columns and "candidate_id" in res_df_check.columns

    if is_candidate_selection:
        matches = _merge_candidate_selection(response_csv, ground_truth_path, join_type)
    elif partial:
        pairs_from_response = merge_response_pairs_partial(response_csv, pairs_path)
        ground_truth_df = load_ground_truth(ground_truth_path)
        matches = merge_dataframes(pairs_from_response, ground_truth_df, join_type=join_type)
    else:
        pairs_from_response = merge_response_pairs(response_csv, pairs_path)
        ground_truth_df = load_ground_truth(ground_truth_path)
        matches = merge_dataframes(pairs_from_response, ground_truth_df, join_type=join_type)

    metrics = evaluate_results(matches)
    
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
