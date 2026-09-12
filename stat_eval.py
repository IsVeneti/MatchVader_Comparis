"""
Statistical comparison of LLM entity matching results.
Integrates with existing evaluation pipeline for per-row correctness.

Usage:
    python stat_eval.py -b path/to/llama path/to/qwen --names llama qwen
    python stat_eval.py -b results/run1 results/run2 results/run3 --names run1 run2 run3
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from pathlib import Path
import yaml

from src.data_processing.pairs_to_ids import get_or_create_id_pairs
from src.evaluator.data_manipulation import (
    load_ground_truth, merge_dataframes,
    merge_response_pairs, merge_response_pairs_partial,
)

DS_CONFIG = "configs/dataset_config.yaml"


# --- Discovery & Loading ---

def discover_results(base_paths, names, pattern="results.csv"):
    """Find all results.csv under each base path, tagged with given name.
    Expected layout per base path: task/dataset/results.csv
    """
    records = []
    for base, name in zip(base_paths, names):
        base = Path(base)
        for p in sorted(base.rglob(pattern)):
            parts = p.relative_to(base).parts  # (task, dataset, "results.csv")
            if len(parts) < 3:
                print(f"  Skipping {p} — not enough path segments")
                continue
            records.append({
                "name": name, "task": parts[0],
                "dataset": parts[1], "path": p,
            })
    print(f"Found {len(records)} result files across {len(names)} sources")
    return pd.DataFrame(records)


def load_correctness(response_csv, dataset_key, config_path=DS_CONFIG,
                     join_type="inner", partial=False):
    """Run evaluation pipeline, return per-row correctness (1/0) indexed by pair_index.

    After merging prediction with ground truth, both have a 'match' column,
    so pandas renames them (e.g. match_x, match_y). Uncomment the debug
    print below on first run to see exact column names.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_conf = config[dataset_key]
    pairs_path = ds_conf["pairs_with_ids"]

    if not Path(pairs_path).exists():
        pid = get_or_create_id_pairs(
            ds_conf["pairs"], ds_conf["d1"], ds_conf["d2"],
            output_csv=pairs_path, force_recreate=False,
        )
        pid.to_csv(pairs_path, index=False)

    if partial:
        merged_resp = merge_response_pairs_partial(str(response_csv), pairs_path)
    else:
        merged_resp = merge_response_pairs(str(response_csv), pairs_path)

    gt_df = load_ground_truth(ds_conf["gt"])
    merged = merge_dataframes(merged_resp, gt_df, join_type=join_type)

    pred_col = "match_pairs"  # suffix set by merge_dataframes(..., suffixes=('_pairs', '_gt'))
    gt_col = "match_gt"
    correct = (merged[pred_col] == merged[gt_col]).astype(int)

    if "pair_index" in merged.columns:
        return pd.Series(correct.values, index=merged["pair_index"], name="correct")
    return correct.reset_index(drop=True)


def build_aligned_table(catalog, name=None, task=None, dataset=None, **kwargs):
    """Load and inner-join correctness across matching conditions."""
    mask = pd.Series(True, index=catalog.index)
    if name is not None:
        mask &= catalog["name"] == name
    if task is not None:
        mask &= catalog["task"] == task
    if dataset is not None:
        mask &= catalog["dataset"] == dataset

    subset = catalog[mask]
    if subset.empty:
        return pd.DataFrame()

    merged = None
    for _, row in subset.iterrows():
        label = f"{row['name']}/{row['task']}"
        series = load_correctness(row["path"], row["dataset"], **kwargs).rename(label)
        if merged is None:
            merged = series.to_frame()
        else:
            merged = merged.join(series, how="inner")

    print(f"  Aligned {len(merged)} examples across {len(merged.columns)} conditions")
    return merged


# --- Statistical Tests ---

def mcnemar_test(col_a, col_b, name_a="A", name_b="B"):
    """Compare two sources' correctness on the same examples."""
    a, b = col_a.values, col_b.values
    n = len(a)

    both_right = np.sum((a == 1) & (b == 1))
    a_only = np.sum((a == 1) & (b == 0))
    b_only = np.sum((a == 0) & (b == 1))
    both_wrong = np.sum((a == 0) & (b == 0))

    print(f"\n--- McNemar: {name_a} vs {name_b} (n={n}) ---")
    print(f"  Both correct: {both_right}  |  Both wrong: {both_wrong}")
    print(f"  Only {name_a}: {a_only}  |  Only {name_b}: {b_only}")
    print(f"  Correct rate — {name_a}: {(both_right+a_only)/n:.3f}, "
          f"{name_b}: {(both_right+b_only)/n:.3f}")

    n_dis = a_only + b_only
    if n_dis == 0:
        print("  No disagreements.")
        return {"p_value": 1.0}

    if n_dis < 25:
        p = stats.binomtest(a_only, n_dis, 0.5).pvalue
        method = "exact binomial"
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / n_dis
        p = 1 - stats.chi2.cdf(chi2, df=1)
        method = f"χ²={chi2:.3f}"

    print(f"  {method}, p={p:.4f} {'**' if p < 0.05 else ''}")
    return {"p_value": p, "a_only": a_only, "b_only": b_only}


def cochrans_q(columns: dict[str, np.ndarray]):
    """Test whether correctness rates differ across k conditions (same examples)."""
    names = list(columns.keys())
    data = np.column_stack([columns[n] for n in names])
    n, k = data.shape

    row_sums = data.sum(axis=1)
    col_sums = data.sum(axis=0)
    N = row_sums.sum()

    num = (k - 1) * (k * np.sum(col_sums**2) - N**2)
    den = k * N - np.sum(row_sums**2)
    if den == 0:
        print("  No variation across conditions.")
        return None

    Q = num / den
    p = 1 - stats.chi2.cdf(Q, df=k - 1)

    print(f"\n--- Cochran's Q (n={n}, k={k}) ---")
    for nm, cs in zip(names, col_sums):
        print(f"  {nm}: {cs}/{n} ({cs/n:.3f})")
    print(f"  Q={Q:.3f}, df={k-1}, p={p:.4f} {'**' if p < 0.05 else ''}")

    if p < 0.05 and k > 2:
        n_comp = k * (k - 1) // 2
        print(f"  Post-hoc ({n_comp} comparisons, Bonferroni):")
        cols_arr = [columns[n] for n in names]
        for i, j in combinations(range(k), 2):
            b = np.sum((cols_arr[i] == 1) & (cols_arr[j] == 0))
            c = np.sum((cols_arr[i] == 0) & (cols_arr[j] == 1))
            if b + c == 0:
                p_pair = 1.0
            elif b + c < 25:
                p_pair = stats.binomtest(b, b + c, 0.5).pvalue
            else:
                chi2_val = (abs(b - c) - 1) ** 2 / (b + c)
                p_pair = 1 - stats.chi2.cdf(chi2_val, df=1)
            p_adj = min(p_pair * n_comp, 1.0)
            print(f"    {names[i]} vs {names[j]}: p={p_adj:.4f} {'**' if p_adj < 0.05 else ''}")

    return {"Q": Q, "p_value": p}


def bootstrap_ci(correct_arr, n_boot=2000, ci=95, seed=42):
    """Bootstrap CI on correctness rate from a single set of outcomes."""
    rng = np.random.default_rng(seed)
    s = np.asarray(correct_arr, dtype=int)
    n = len(s)
    rates = np.array([rng.choice(s, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(rates, [alpha, 100 - alpha])
    return {"correct_rate": s.mean(), "ci_low": lo, "ci_high": hi}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of LLM entity matching results"
    )
    parser.add_argument(
        "-b", "--base_paths", nargs="+", required=True,
        help="Base directories to compare (each contains task/dataset/results.csv)"
    )
    parser.add_argument(
        "-n", "--names", nargs="+", required=True,
        help="Labels for each base path (must match count of --base_paths)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Task names to include (default: auto-detect)"
    )
    parser.add_argument(
        "--config", default=DS_CONFIG,
        help="Path to dataset config YAML"
    )
    parser.add_argument(
        "--join", default="inner", choices=["inner", "outer", "left", "right"],
    )
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    if len(args.base_paths) != len(args.names):
        parser.error("--base_paths and --names must have the same count")

    eval_kwargs = dict(config_path=args.config, join_type=args.join, partial=args.partial)

    catalog = discover_results(args.base_paths, args.names)
    print(catalog[["name", "task", "dataset"]].to_string(index=False))

    tasks = args.tasks or sorted(catalog["task"].unique())
    names = args.names
    datasets = sorted(catalog["dataset"].unique())

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds}")
        print(f"{'='*60}")

        # McNemar: pairwise comparison of sources per task
        for task in tasks:
            table = build_aligned_table(catalog, task=task, dataset=ds, **eval_kwargs)
            if table.empty or len(table.columns) < 2:
                continue
            cols = list(table.columns)
            for i, j in combinations(range(len(cols)), 2):
                mcnemar_test(table[cols[i]], table[cols[j]], cols[i], cols[j])

        # Cochran's Q: compare tasks within each source
        for name in names:
            table = build_aligned_table(catalog, name=name, dataset=ds, **eval_kwargs)
            if table.empty or len(table.columns) < 2:
                continue
            print(f"\n>>> Cochran's Q for {name}")
            cochrans_q({c: table[c].values for c in table.columns})

        # Bootstrap CIs
        print(f"\n--- Bootstrap 95% CIs (correctness) ---")
        for name in names:
            for task in tasks:
                sub = catalog[
                    (catalog["name"] == name) &
                    (catalog["task"] == task) &
                    (catalog["dataset"] == ds)
                ]
                if sub.empty:
                    continue
                correct = load_correctness(sub.iloc[0]["path"], ds, **eval_kwargs)
                r = bootstrap_ci(correct.values)
                print(f"  {name}/{task}: "
                      f"{r['correct_rate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


if __name__ == "__main__":
    main()