import argparse
from pathlib import Path
import pandas as pd
from evaluate_main import evaluate_file

def find_all_results(base_path, pattern="results.csv"):
    """Find all results.csv files in the directory structure."""
    base_path = Path(base_path)
    return sorted(base_path.rglob(pattern))

def extract_metadata(results_path, base_path):
    """Extract dataset, date, and taskname from path."""
    rel_path = results_path.relative_to(base_path)
    parts = rel_path.parts
    
    return {
        'dataset': parts[0] if len(parts) > 0 else 'unknown',
        'date': parts[1] if len(parts) > 1 else 'unknown',
        'taskname': parts[2] if len(parts) > 2 else 'unknown',
        'full_path': str(results_path)
    }

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate all results')
    parser.add_argument('--base_path', '-b', default='results', 
                       help='Base path to search for results')
    parser.add_argument('--pattern', default='results.csv', 
                       help='Filename pattern to search for')
    parser.add_argument('--join', '-j', choices=['inner', 'outer', 'left', 'right'], 
                       default='inner', help='Join type for comparison')
    parser.add_argument('--partial', '-p', action='store_true',
                       help='Handle partial runs')
    parser.add_argument('--output', '-o', default='batch_evaluation_results.csv',
                       help='Output CSV file for aggregated results')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print what would be evaluated without running')
    
    args = parser.parse_args()
    
    # Find all results files
    results_files = find_all_results(args.base_path, args.pattern)
    
    if not results_files:
        print(f"No {args.pattern} files found in {args.base_path}")
        return
    
    print(f"Found {len(results_files)} results files to evaluate\n")
    
    # Process each file
    all_results = []
    
    for i, results_path in enumerate(results_files, 1):
        metadata = extract_metadata(results_path, args.base_path)
        
        print(f"[{i}/{len(results_files)}] Processing: {metadata['dataset']}/{metadata['date']}/{metadata['taskname']}")
        
        if args.dry_run:
            print(f"  Would evaluate: {results_path}")
            continue
        
        try:
            # Call the evaluation function directly
            metrics = evaluate_file(
                dataset=metadata['dataset'],
                response_csv=str(results_path),
                join_type=args.join,
                partial=args.partial
            )
            
            print("  ✓ Evaluation completed")
            
            result_record = {
                **metadata,
                **metrics,
                'eval_status': 'success'
            }
            
        except Exception as e:
            print(f"  ✗ Evaluation failed: {str(e)}")
            result_record = {
                **metadata,
                'eval_status': 'failed',
                'error': str(e)
            }
        
        all_results.append(result_record)
        print()
    
    if args.dry_run:
        print("Dry run complete. No evaluations were run.")
        return
    
    # Save aggregated results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"Aggregated results saved to: {args.output}")
    
    # Print summary
    success_count = sum(1 for r in all_results if r['eval_status'] == 'success')
    print(f"\nSummary:")
    print(f"  Total: {len(all_results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(all_results) - success_count}")

if __name__ == '__main__':
    main()