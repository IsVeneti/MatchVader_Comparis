import argparse
from src.evaluator.data_manipulation import filter_csv_columns, merge_dataframes, merge_response_pairs
from src.evaluator.evaluate import evaluate_results
from src.data_processing.pairs_to_ids import get_or_create_id_pairs
import pandas as pd

def run_evaluation(pairs_with_ids, gt_csv, join_type='inner', output_csv=None):
    """Run complete evaluation pipeline."""
    
    # Compare with ground truth
    matches = merge_dataframes(pairs_with_ids, gt_csv, join_type=join_type)
    
    # Calculate metrics
    evaluate_results(matches)
    
    # # Save if specified
    # if output_csv:
    #     matches.to_csv(output_csv, index=False)
    #     print(f"\n✓ Saved matches to '{output_csv}'")
    
    # return results, matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate candidate pairs against ground truth')
    
    # Required arguments
    parser.add_argument('--pairs', '-p',help='Candidate pairs CSV file')
    # parser.add_argument('--dataset1','-d2', help='First dataset CSV file', required=True)
    # parser.add_argument('--dataset2','-d2', help='Second dataset CSV file',required=True)
    parser.add_argument('--ground_truth','-gt', help='Ground truth CSV file', required=True)
    parser.add_argument('--response_csv','-r', help='LLM response CSV file', required=True)
    
    # Optional arguments
    parser.add_argument('--join', '-j', 
                       choices=['inner', 'outer', 'left', 'right'],
                       default='inner',
                       help='Join type for comparison (default: inner)')
    
    parser.add_argument('--output', '-o',
                       help='Output CSV file for matches (optional)')
    
    
    # parser.add_argument('--pairs_clean', '-po',
    #                    default='d1_pairs_with_ids.csv',
    #                    help='Cache file for ID-based pairs (default: d1_pairs_with_ids.csv)')
    
    args = parser.parse_args()
    # This currently doesn't work because i used the pairs with indexes to run the LLM
    # TODO: Fix this
    # pairs_from_response = filter_csv_columns(args.response_csv, columns=['id1', 'id2', 'match'])

    pairs_from_response = merge_response_pairs(args.response_csv, args.pairs)
    ground_truth_df = pd.read_csv(args.ground_truth, sep=',')

    # Run evaluation
    run_evaluation(
        pairs_from_response,
        ground_truth_df,
        join_type=args.join,
        output_csv=args.output
    )
    
    # # Print results
    # print(f"\n=== EVALUATION RESULTS ({args.join.upper()} JOIN) ===")
    # for key, value in results.items():
    #     if isinstance(value, float):
    #         print(f"{key}: {value:.2%}" if 'precision' in key or 'recall' in key or 'f1' in key else f"{key}: {value:.4f}")
    #     else:
    #         print(f"{key}: {value}")