import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_main import evaluate_file

LAYOUTS = {
    'legacy': {
        'description': 'dataset/date/taskname/results.csv',
        'fields': ('dataset', 'date', 'taskname'),
    },
    'named': {
        'description': 'task/dataset_{number}/results.csv',
        'fields': ('task', 'dataset'),
    }
}


def find_all_results(base_path, pattern="results.csv"):
    """Find all results.csv files in the directory structure."""
    base_path = Path(base_path)
    return sorted(base_path.rglob(pattern))


def extract_metadata(results_path, base_path, layout='legacy'):
    """Extract metadata from path based on the chosen layout.

    Layouts:
        legacy: dataset/date/taskname/results.csv
        named:  name/task/dataset_{number}/results.csv
    """
    rel_path = results_path.relative_to(base_path)
    parts = rel_path.parts

    fields = LAYOUTS[layout]['fields']
    metadata = {
        field: parts[i] if i < len(parts) - 1 else 'unknown'  # -1 to skip filename
        for i, field in enumerate(fields)
    }
    metadata['full_path'] = str(results_path)

    # For the named layout, also parse out the number suffix if present
    if layout == 'named' and 'dataset' in metadata and metadata['dataset'] != 'unknown':
        ds = metadata['dataset']
        if '_' in ds:
            prefix, _, num = ds.rpartition('_')
            if num.isdigit():
                metadata['dataset_prefix'] = prefix
                metadata['dataset_number'] = int(num)

    return metadata


def load_run_metadata(results_path, metadata_filename="metadata.json"):
    """Load duration and token usage from a metadata.json file next to results.csv.

    Returns a flat dict with 'duration_minutes' and all token_usage fields,
    or an empty dict if the file is missing or malformed.
    """
    metadata_path = results_path.parent / metadata_filename
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠ Could not read {metadata_path}: {e}")
        return {}

    run_meta = {}

    duration = meta.get("run_info", {}).get("duration_minutes")
    if duration is not None:
        run_meta["duration_minutes"] = duration

    token_usage = meta.get("token_usage", {})
    for key, value in token_usage.items():
        run_meta[f"token_{key}"] = value  # e.g. token_prompt_tokens, token_llm_calls

    return run_meta


def format_label(metadata, layout):
    """Build a short display label from metadata."""
    fields = LAYOUTS[layout]['fields']
    return '/'.join(metadata.get(f, '?') for f in fields)


def main():
    layout_help = ' | '.join(
        f"'{k}': {v['description']}" for k, v in LAYOUTS.items()
    )

    parser = argparse.ArgumentParser(description='Batch evaluate all results')
    parser.add_argument('--base_path', '-b', default='results',
                        help='Base path to search for results')
    parser.add_argument('--pattern', default='results.csv',
                        help='Filename pattern to search for')
    parser.add_argument('--layout', '-l', choices=LAYOUTS.keys(), default='named',
                        help=f'Directory layout to use ({layout_help})')
    parser.add_argument('--join', '-j', choices=['inner', 'outer', 'left', 'right'],
                        default='left', help='Join type for comparison')
    parser.add_argument('--partial', '-p', action='store_true',
                        help='Handle partial runs')
    parser.add_argument('--output', '-o', default='batch_evaluation_results.csv',
                        help='Output CSV file for aggregated results')
    parser.add_argument('--metadata_file', default='metadata.json',
                        help='Metadata filename to look for alongside each results file')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be evaluated without running')

    args = parser.parse_args()

    # Find all results files
    results_files = find_all_results(args.base_path, args.pattern)

    if not results_files:
        print(f"No {args.pattern} files found in {args.base_path}")
        return

    print(f"Found {len(results_files)} results files to evaluate")
    print(f"Using layout: {args.layout} ({LAYOUTS[args.layout]['description']})\n")

    # Process each file
    all_results = []

    for i, results_path in enumerate(results_files, 1):
        metadata = extract_metadata(results_path, args.base_path, layout=args.layout)
        label = format_label(metadata, args.layout)

        print(f"[{i}/{len(results_files)}] Processing: {label}")

        if args.dry_run:
            print(f"  Would evaluate: {results_path}")
            continue

        # Load run metadata (duration + token usage)
        run_meta = load_run_metadata(results_path, args.metadata_file)
        if run_meta:
            print(f"  ✓ Loaded run metadata ({len(run_meta)} fields)")
        else:
            print(f"  ⚠ No run metadata found")

        try:
            dataset_key = metadata.get('dataset', 'unknown')

            metrics = evaluate_file(
                dataset=dataset_key,
                response_csv=str(results_path),
                join_type=args.join,
                partial=args.partial
            )

            print("  ✓ Evaluation completed")

            result_record = {
                **metadata,
                **run_meta,
                **metrics,
                'eval_status': 'success'
            }

        except Exception as e:
            print(f"  ✗ Evaluation failed: {str(e)}")
            result_record = {
                **metadata,
                **run_meta,
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