"""
Statistical analysis of LLM entity matching results.

Reads raw results.csv files from:
    C:\\Matchvader_results\\yoda results\\<model>\\<task>\\<dataset>\\results.csv

Uses the existing evaluation pipeline (merge_response_pairs + merge_dataframes)
to obtain per-row predictions vs ground truth, then runs:

  1. Class imbalance summary (positive rate, neg/pos ratio, MCC)
  2. McNemar test — model vs model (llama_yoda vs qwen), per task × dataset
  3. McNemar test — task vs task (all pairs), per model × dataset
  4. McNemar test — per task (collapsed across datasets, for an overall view)
  5. Bootstrap 95% CIs on F1, accuracy, precision, recall
  6. Error analysis (FP, FN, FPR, FNR per condition × dataset)

Output written to analysis_output/:
    stat_results.txt          human-readable report
    class_imbalance.csv
    mcnemar_model.csv
    mcnemar_task.csv
    mcnemar_task_overall.csv
    bootstrap_ci.csv
    error_analysis.csv

Usage:
    python stat_analysis.py
    python stat_analysis.py --base "C:\\Matchvader_results\\yoda results" --n-boot 2000
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.evaluator.data_manipulation import (
    load_ground_truth,
    merge_dataframes,
    merge_response_pairs,
)

DS_CONFIG = "configs/dataset_config.yaml"
DEFAULT_BASE = r"C:\Matchvader_results\yoda results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(base: Path, config: dict, model_suffix: str = "") -> pd.DataFrame:
    """
    Walk base/<model>/<task>/<dataset>/results.csv, run the evaluation pipeline,
    and return a single DataFrame with columns:
        model, task, dataset_number, id1, id2, y_pred, y_true, correct

    model_suffix: appended to each model name, e.g. "_rep" to distinguish a
                  repetition-prompt run from the base run when both are loaded together.
    """
    records = []
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name + model_suffix
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for ds_dir in sorted(task_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                ds_key = ds_dir.name          # e.g. "dataset_1"
                results_csv = ds_dir / "results.csv"
                if not results_csv.exists():
                    print(f"  SKIP (no results.csv): {results_csv}")
                    continue
                if ds_key not in config:
                    print(f"  SKIP (not in config): {ds_key}")
                    continue

                ds_num = int(ds_key.split("_")[1])
                pairs_path = config[ds_key]["pairs_with_ids"]
                gt_path    = config[ds_key]["gt"]

                try:
                    gt_df = load_ground_truth(gt_path)
                    is_candidate = task.startswith("CandidateSelection")

                    if is_candidate:
                        # Candidate selection results already have target_id/candidate_id columns;
                        # merge directly on those instead of relying on row-order alignment.
                        res_df = pd.read_csv(results_csv, encoding="utf-8")
                        if "target_id" not in res_df.columns or "candidate_id" not in res_df.columns:
                            raise ValueError("Missing target_id/candidate_id columns in candidate selection results")

                        metadata_path = results_csv.parent / "metadata.json"
                        if not metadata_path.exists():
                            raise FileNotFoundError(f"metadata.json not found next to {results_csv}")
                        with open(metadata_path) as mf:
                            meta = json.load(mf)
                        target_side = meta["task"]["config"]["target_side"]  # "d1" or "d2"

                        if target_side == "d1":
                            # D1Target: target=d1->id1, candidate=d2->id2
                            res_df = res_df[["target_id", "candidate_id", "match"]].rename(
                                columns={"target_id": "id1", "candidate_id": "id2", "match": "match_pairs"}
                            )
                        else:
                            # D2Target: target=d2->id2, candidate=d1->id1
                            res_df = res_df[["candidate_id", "target_id", "match"]].rename(
                                columns={"candidate_id": "id1", "target_id": "id2", "match": "match_pairs"}
                            )

                        gt_renamed = gt_df.rename(columns={gt_df.columns[0]: "id1", gt_df.columns[1]: "id2",
                                                            "match": "match_gt"})
                        merged = res_df.merge(gt_renamed[["id1", "id2", "match_gt"]],
                                              on=["id1", "id2"], how="left")
                        merged["match_pairs"] = pd.to_numeric(merged["match_pairs"], errors="coerce").fillna(0).astype(int)
                        merged["match_gt"] = pd.to_numeric(merged["match_gt"], errors="coerce").fillna(0).astype(int)
                    else:
                        merged_resp = merge_response_pairs(str(results_csv), pairs_path)
                        # Left join: keep all predicted pairs; non-GT pairs get match_gt=0
                        merged = merge_dataframes(merged_resp, gt_df, join_type="left")
                except Exception as e:
                    print(f"  ERROR {model}/{task}/{ds_key}: {e}")
                    continue

                # Pipeline yields match_pairs (prediction) and match_gt (ground truth)
                y_pred = merged["match_pairs"].values
                y_true = merged["match_gt"].values

                df_part = pd.DataFrame({
                    "model":          model,
                    "task":           task,
                    "dataset_number": ds_num,
                    "id1":            merged["id1"].values,
                    "id2":            merged["id2"].values,
                    "y_pred":         y_pred,
                    "y_true":         y_true,
                    "correct":        (y_pred == y_true).astype(int),
                })
                records.append(df_part)
                print(f"  OK  {model}/{task}/{ds_key}: {len(df_part)} rows")

    if not records:
        sys.exit("No results loaded — check --base path and dataset config.")

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def binary_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    n  = len(y_true)
    acc  = (tp + tn) / n if n > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=acc, precision=prec, recall=rec, f1=f1, mcc=mcc)


def bootstrap_metric(y_true, y_pred, metric="f1", n_boot=2000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(binary_metrics(y_true[idx], y_pred[idx])[metric])
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boots, [alpha, 100 - alpha])
    return {"point": binary_metrics(y_true, y_pred)[metric], "ci_low": lo, "ci_high": hi}


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------

def mcnemar(correct_a, correct_b, name_a="A", name_b="B"):
    """
    McNemar's test on paired correctness arrays.
    Uses exact binomial for n_discordant < 25, continuity-corrected χ² otherwise.
    """
    a, b = np.asarray(correct_a), np.asarray(correct_b)
    n = len(a)
    both_right = int(np.sum((a == 1) & (b == 1)))
    a_only     = int(np.sum((a == 1) & (b == 0)))
    b_only     = int(np.sum((a == 0) & (b == 1)))
    both_wrong = int(np.sum((a == 0) & (b == 0)))
    n_dis = a_only + b_only

    if n_dis == 0:
        return dict(name_a=name_a, name_b=name_b, n=n,
                    both_right=both_right, both_wrong=both_wrong,
                    a_only=a_only, b_only=b_only,
                    rate_a=(both_right+a_only)/n, rate_b=(both_right+b_only)/n,
                    method="no_disagreements", statistic=np.nan, p_value=1.0, significant=False)

    if n_dis < 25:
        p    = stats.binomtest(a_only, n_dis, 0.5).pvalue
        method, stat = "exact_binomial", np.nan
    else:
        stat = (abs(a_only - b_only) - 1) ** 2 / n_dis
        p    = 1 - stats.chi2.cdf(stat, df=1)
        method = "chi2_cc"

    return dict(name_a=name_a, name_b=name_b, n=n,
                both_right=both_right, both_wrong=both_wrong,
                a_only=a_only, b_only=b_only,
                rate_a=(both_right+a_only)/n, rate_b=(both_right+b_only)/n,
                method=method, statistic=stat, p_value=p, significant=(p < 0.05))


def apply_bonferroni(df: pd.DataFrame, n_comp: int) -> pd.DataFrame:
    df = df.copy()
    df["p_bonferroni"]   = (df["p_value"] * n_comp).clip(upper=1.0)
    df["sig_bonferroni"] = df["p_bonferroni"] < 0.05
    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def class_imbalance_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, ds_num), grp in df.groupby(["model", "task", "dataset_number"]):
        m = binary_metrics(grp["y_true"], grp["y_pred"])
        n_pos = m["tp"] + m["fn"]
        n_neg = m["fp"] + m["tn"]
        rows.append(dict(
            model=model, task=task, dataset_number=ds_num,
            n=m["n"], n_pos=n_pos, n_neg=n_neg,
            pos_rate=round(n_pos/m["n"], 4) if m["n"] > 0 else np.nan,
            imb_ratio=round(n_neg/n_pos, 2) if n_pos > 0 else np.inf,
            f1=round(m["f1"], 4), mcc=round(m["mcc"], 4),
            accuracy=round(m["accuracy"], 4),
            precision=round(m["precision"], 4),
            recall=round(m["recall"], 4),
        ))
    return pd.DataFrame(rows)


def mcnemar_model_vs_model(df: pd.DataFrame) -> pd.DataFrame:
    """llama_yoda vs qwen, per task × dataset, paired on (id1, id2)."""
    models = sorted(df["model"].unique())
    if len(models) < 2:
        return pd.DataFrame()

    records = []
    for (task, ds_num), grp in df.groupby(["task", "dataset_number"]):
        # Build per-model series indexed by (id1, id2)
        model_series = {}
        for model in models:
            sub = grp[grp["model"] == model].set_index(["id1", "id2"])["correct"]
            model_series[model] = sub

        for m_a, m_b in combinations(models, 2):
            common = model_series[m_a].index.intersection(model_series[m_b].index)
            if len(common) < 5:
                continue
            res = mcnemar(
                model_series[m_a].loc[common].values,
                model_series[m_b].loc[common].values,
                name_a=m_a, name_b=m_b,
            )
            res.update({"task": task, "dataset_number": ds_num})
            records.append(res)

    result = pd.DataFrame(records)
    if not result.empty and len(result) > 1:
        result = apply_bonferroni(result, len(result))
    return result


def mcnemar_task_vs_task(df: pd.DataFrame) -> pd.DataFrame:
    """All task pairs, per model × dataset, paired on (id1, id2)."""
    tasks = sorted(df["task"].unique())
    if len(tasks) < 2:
        return pd.DataFrame()

    records = []
    for (model, ds_num), grp in df.groupby(["model", "dataset_number"]):
        task_series = {}
        for task in tasks:
            sub = grp[grp["task"] == task].set_index(["id1", "id2"])["correct"]
            task_series[task] = sub

        for t_a, t_b in combinations(tasks, 2):
            if t_a not in task_series or t_b not in task_series:
                continue
            common = task_series[t_a].index.intersection(task_series[t_b].index)
            if len(common) < 5:
                continue
            res = mcnemar(
                task_series[t_a].loc[common].values,
                task_series[t_b].loc[common].values,
                name_a=t_a, name_b=t_b,
            )
            res.update({"model": model, "dataset_number": ds_num})
            records.append(res)

    result = pd.DataFrame(records)
    if not result.empty and len(result) > 1:
        result = apply_bonferroni(result, len(result))
    return result


def mcnemar_task_overall(df: pd.DataFrame) -> pd.DataFrame:
    """
    McNemar per task pair, collapsed across all datasets for each model.
    Pairs examples across datasets by concatenating — gives an overall
    signal rather than per-dataset.
    """
    tasks = sorted(df["task"].unique())
    models = sorted(df["model"].unique())
    if len(tasks) < 2:
        return pd.DataFrame()

    records = []
    for model in models:
        m_grp = df[df["model"] == model]
        # Use a globally unique key: (dataset_number, id1, id2)
        task_series = {}
        for task in tasks:
            sub = (m_grp[m_grp["task"] == task]
                   .set_index(["dataset_number", "id1", "id2"])["correct"])
            task_series[task] = sub

        for t_a, t_b in combinations(tasks, 2):
            if t_a not in task_series or t_b not in task_series:
                continue
            common = task_series[t_a].index.intersection(task_series[t_b].index)
            if len(common) < 5:
                continue
            res = mcnemar(
                task_series[t_a].loc[common].values,
                task_series[t_b].loc[common].values,
                name_a=t_a, name_b=t_b,
            )
            res.update({"model": model})
            records.append(res)

    result = pd.DataFrame(records)
    if not result.empty and len(result) > 1:
        result = apply_bonferroni(result, len(result))
    return result


def bootstrap_all(df: pd.DataFrame, n_boot=2000, seed=42) -> pd.DataFrame:
    records = []
    for (model, task, ds_num), grp in df.groupby(["model", "task", "dataset_number"]):
        for metric in ("f1", "accuracy", "precision", "recall", "mcc"):
            ci = bootstrap_metric(grp["y_true"], grp["y_pred"],
                                  metric=metric, n_boot=n_boot, seed=seed)
            records.append(dict(
                model=model, task=task, dataset_number=ds_num,
                n=len(grp), metric=metric,
                point=round(ci["point"], 4),
                ci_low=round(ci["ci_low"], 4),
                ci_high=round(ci["ci_high"], 4),
            ))
    return pd.DataFrame(records)


def error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, ds_num), grp in df.groupby(["model", "task", "dataset_number"]):
        m = binary_metrics(grp["y_true"], grp["y_pred"])
        n_pos = m["tp"] + m["fn"]
        n_neg = m["fp"] + m["tn"]
        fpr = m["fp"] / n_neg if n_neg > 0 else np.nan
        fnr = m["fn"] / n_pos if n_pos > 0 else np.nan
        err_type = ("FP-heavy" if m["fp"] > m["fn"] else
                    "FN-heavy" if m["fn"] > m["fp"] else "balanced")
        rows.append(dict(
            model=model, task=task, dataset_number=ds_num, n=m["n"],
            n_actual_pos=n_pos, n_actual_neg=n_neg,
            tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
            fpr=round(fpr, 4), fnr=round(fnr, 4),
            f1=round(m["f1"], 4), mcc=round(m["mcc"], 4),
            error_type=err_type,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def section(title):
    bar = "=" * 72
    return f"\n{bar}\n  {title}\n{bar}\n"


def fmt_mcnemar(mdf: pd.DataFrame, group_col=None) -> str:
    if mdf.empty:
        return "  (no results)\n"
    display_cols = ["dataset_number", "n", "rate_a", "rate_b",
                    "a_only", "b_only", "method", "p_value", "significant"]
    if "p_bonferroni" in mdf.columns:
        display_cols += ["p_bonferroni", "sig_bonferroni"]
    if group_col:
        display_cols = [group_col] + display_cols
    available = [c for c in display_cols if c in mdf.columns]

    lines = []
    if group_col and group_col in mdf.columns:
        for key, sub in mdf.groupby(group_col):
            lines.append(f"\n  [{group_col}={key}]\n")
            lines.append(
                sub[available].sort_values("dataset_number")
                .to_string(index=False) + "\n"
            )
    else:
        lines.append(mdf[available].to_string(index=False) + "\n")
    return "".join(lines)


def fmt_bootstrap(bdf: pd.DataFrame, metric: str) -> str:
    sub = bdf[bdf["metric"] == metric].copy()
    sub["ci"] = sub.apply(lambda r: f"[{r.ci_low:.4f}, {r.ci_high:.4f}]", axis=1)
    cols = ["model", "task", "dataset_number", "point", "ci", "n"]
    return sub[cols].to_string(index=False) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of entity matching results")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Root directory: base/<model>/<task>/<dataset>/results.csv")
    parser.add_argument("--extra-base", default=None,
                        help="Additional root directory to load alongside --base (models get --extra-suffix appended)")
    parser.add_argument("--extra-suffix", default="_rep",
                        help="Suffix appended to model names from --extra-base (default: '_rep')")
    parser.add_argument("--config", default=DS_CONFIG)
    parser.add_argument("--output-dir", default="analysis_output")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--excel", action="store_true",
                        help="Also write an Excel workbook with one sheet per CSV")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        sys.exit(f"Base path not found: {base}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    print("\nLoading results...")
    df = load_all_results(base, config)
    if args.extra_base:
        extra_base = Path(args.extra_base)
        if not extra_base.exists():
            sys.exit(f"Extra base path not found: {extra_base}")
        print(f"\nLoading extra results from {extra_base} (suffix='{args.extra_suffix}')...")
        df_extra = load_all_results(extra_base, config, model_suffix=args.extra_suffix)
        df = pd.concat([df, df_extra], ignore_index=True)
    print(f"\nTotal rows loaded: {len(df):,}")
    print(f"Models:   {sorted(df['model'].unique())}")
    print(f"Tasks:    {sorted(df['task'].unique())}")
    print(f"Datasets: {sorted(df['dataset_number'].unique())}")

    lines = []
    lines.append("Statistical Analysis Report — MatchVader Entity Matching\n")
    lines.append(f"Base:  {base}\n")
    lines.append(f"Rows:  {len(df):,}\n")
    lines.append(f"Models:  {sorted(df['model'].unique())}\n")
    lines.append(f"Tasks:   {sorted(df['task'].unique())}\n")
    lines.append(f"Datasets: {sorted(df['dataset_number'].unique())}\n")

    total_pos = int(df["y_true"].sum())
    total_n   = len(df)
    lines.append(f"\nOverall positive rate: {total_pos}/{total_n} = {total_pos/total_n:.3f}\n")
    lines.append(f"Overall imbalance ratio (neg/pos): {(total_n-total_pos)/total_pos:.2f}\n")

    # ------------------------------------------------------------------
    # 1. Class imbalance
    # ------------------------------------------------------------------
    print("\nComputing class imbalance summary...")
    imb = class_imbalance_summary(df)
    imb.to_csv(out_dir / "class_imbalance.csv", index=False)

    lines.append(section("1. CLASS IMBALANCE & PER-CONDITION METRICS"))
    lines.append(
        "Columns: pos_rate = fraction of positive pairs, "
        "imb_ratio = neg/pos, MCC = Matthews Correlation Coefficient.\n"
        "MCC is robust to class imbalance: 1.0 = perfect, 0 = random, -1 = inverse.\n\n"
    )

    # Per-dataset overview (aggregated across models and tasks)
    agg = df.groupby("dataset_number").agg(
        n=("y_true","count"),
        n_pos=("y_true","sum"),
    ).reset_index()
    agg["n_neg"]    = agg["n"] - agg["n_pos"]
    agg["pos_rate"] = (agg["n_pos"] / agg["n"]).round(4)
    agg["imb_ratio"]= (agg["n_neg"] / agg["n_pos"]).round(2)
    lines.append("Per dataset (all models and tasks combined):\n")
    lines.append(agg.to_string(index=False) + "\n\n")

    lines.append("Per model × task × dataset:\n")
    lines.append(
        imb[["model","task","dataset_number","n","pos_rate","imb_ratio",
             "accuracy","precision","recall","f1","mcc"]]
        .to_string(index=False) + "\n"
    )

    # ------------------------------------------------------------------
    # 2. McNemar: model vs model
    # ------------------------------------------------------------------
    print("\nRunning McNemar: model vs model...")
    mc_model = mcnemar_model_vs_model(df)
    mc_model.to_csv(out_dir / "mcnemar_model.csv", index=False)

    lines.append(section("2. McNEMAR TEST — MODEL vs MODEL (llama_yoda vs qwen)"))
    lines.append(
        "Paired on (id1, id2) within each task × dataset.\n"
        "rate_a = correctness of name_a, rate_b = correctness of name_b.\n"
        "a_only = examples where only A was correct; b_only = only B.\n"
        "Bonferroni correction applied across all comparisons.\n\n"
    )

    if mc_model.empty:
        lines.append("  (only one model found — no comparison)\n")
    else:
        for task in sorted(mc_model["task"].unique()):
            lines.append(f"\n  Task: {task}\n")
            sub = mc_model[mc_model["task"] == task]
            cols = ["dataset_number","n","rate_a","rate_b","a_only","b_only",
                    "method","p_value","significant"]
            if "p_bonferroni" in sub.columns:
                cols += ["p_bonferroni","sig_bonferroni"]
            lines.append(sub[cols].sort_values("dataset_number").to_string(index=False) + "\n")

        sig   = mc_model["significant"].sum()
        total = len(mc_model)
        lines.append(f"\nSignificant (raw p<0.05): {sig}/{total}\n")
        if "sig_bonferroni" in mc_model.columns:
            lines.append(f"Significant (Bonferroni): {mc_model['sig_bonferroni'].sum()}/{total}\n")

    # ------------------------------------------------------------------
    # 3. McNemar: task vs task (per dataset)
    # ------------------------------------------------------------------
    print("\nRunning McNemar: task vs task (per dataset)...")
    mc_task = mcnemar_task_vs_task(df)
    mc_task.to_csv(out_dir / "mcnemar_task.csv", index=False)

    lines.append(section("3. McNEMAR TEST — TASK vs TASK (per model × dataset)"))
    lines.append(
        "All pairwise task comparisons, paired on (id1, id2).\n"
        "Only examples present in both tasks are compared (inner join).\n\n"
    )

    if mc_task.empty:
        lines.append("  (fewer than 2 tasks or no overlapping pairs)\n")
    else:
        for model in sorted(mc_task["model"].unique()):
            lines.append(f"\n  Model: {model}\n")
            sub = mc_task[mc_task["model"] == model]
            for (t_a, t_b), sub2 in sub.groupby(["name_a", "name_b"]):
                lines.append(f"    {t_a}  vs  {t_b}\n")
                cols = ["dataset_number","n","rate_a","rate_b","a_only","b_only",
                        "method","p_value","significant"]
                if "p_bonferroni" in sub2.columns:
                    cols += ["p_bonferroni","sig_bonferroni"]
                lines.append(sub2[cols].sort_values("dataset_number").to_string(index=False) + "\n\n")

        sig   = mc_task["significant"].sum()
        total = len(mc_task)
        lines.append(f"Significant (raw p<0.05): {sig}/{total}\n")
        if "sig_bonferroni" in mc_task.columns:
            lines.append(f"Significant (Bonferroni): {mc_task['sig_bonferroni'].sum()}/{total}\n")

    # ------------------------------------------------------------------
    # 4. McNemar: task vs task overall (collapsed across datasets)
    # ------------------------------------------------------------------
    print("\nRunning McNemar: task vs task (overall)...")
    mc_overall = mcnemar_task_overall(df)
    mc_overall.to_csv(out_dir / "mcnemar_task_overall.csv", index=False)

    lines.append(section("4. McNEMAR TEST — TASK vs TASK (overall, collapsed across datasets)"))
    lines.append(
        "Same as Section 3 but pools all datasets together, giving a single\n"
        "overall comparison per model. Uses (dataset_number, id1, id2) as the key.\n\n"
    )

    if mc_overall.empty:
        lines.append("  (no results)\n")
    else:
        for model in sorted(mc_overall["model"].unique()):
            lines.append(f"\n  Model: {model}\n")
            sub = mc_overall[mc_overall["model"] == model]
            cols = ["name_a","name_b","n","rate_a","rate_b","a_only","b_only",
                    "method","p_value","significant"]
            if "p_bonferroni" in sub.columns:
                cols += ["p_bonferroni","sig_bonferroni"]
            lines.append(sub[cols].to_string(index=False) + "\n")

        sig   = mc_overall["significant"].sum()
        total = len(mc_overall)
        lines.append(f"\nSignificant (raw p<0.05): {sig}/{total}\n")
        if "sig_bonferroni" in mc_overall.columns:
            lines.append(f"Significant (Bonferroni): {mc_overall['sig_bonferroni'].sum()}/{total}\n")

    # ------------------------------------------------------------------
    # 5. Bootstrap CIs
    # ------------------------------------------------------------------
    print("\nComputing bootstrap CIs...")
    boot = bootstrap_all(df, n_boot=args.n_boot, seed=args.seed)
    boot.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    lines.append(section(f"5. BOOTSTRAP 95% CIs  (n_boot={args.n_boot})"))
    lines.append("Point estimate ± 95% percentile bootstrap CI per metric.\n")

    for metric in ("f1", "accuracy", "precision", "recall", "mcc"):
        lines.append(f"\n  Metric: {metric.upper()}\n")
        lines.append(fmt_bootstrap(boot, metric))

    # ------------------------------------------------------------------
    # 6. Error analysis
    # ------------------------------------------------------------------
    print("\nRunning error analysis...")
    err = error_analysis(df)
    err.to_csv(out_dir / "error_analysis.csv", index=False)

    lines.append(section("6. ERROR ANALYSIS"))
    lines.append(
        "FPR = FP / actual negatives  (false alarm rate)\n"
        "FNR = FN / actual positives  (miss rate)\n"
        "error_type: FP-heavy | FN-heavy | balanced\n\n"
    )

    lines.append(err[["model","task","dataset_number","n",
                       "n_actual_pos","n_actual_neg",
                       "tp","fp","fn","tn","fpr","fnr","f1","mcc","error_type"]]
                 .to_string(index=False) + "\n")

    lines.append("\n  Error type summary:\n")
    lines.append(err["error_type"].value_counts().to_string() + "\n")

    lines.append("\n  Per-model totals (all tasks and datasets):\n")
    for model in sorted(err["model"].unique()):
        sub = err[err["model"] == model]
        lines.append(
            f"    {model}: TP={sub['tp'].sum():,}  FP={sub['fp'].sum():,}  "
            f"FN={sub['fn'].sum():,}  TN={sub['tn'].sum():,}\n"
        )

    lines.append("\n  Per-task totals (all models and datasets):\n")
    for task in sorted(err["task"].unique()):
        sub = err[err["task"] == task]
        lines.append(
            f"    {task}: TP={sub['tp'].sum():,}  FP={sub['fp'].sum():,}  "
            f"FN={sub['fn'].sum():,}  TN={sub['tn'].sum():,}\n"
        )

    lines.append("\n  5 worst FNR (highest miss rate for actual positives):\n")
    lines.append(
        err.nlargest(5, "fnr")[["model","task","dataset_number","fnr","fpr","f1","n_actual_pos"]]
        .to_string(index=False) + "\n"
    )

    lines.append("\n  5 worst FPR (highest false alarm rate on negatives):\n")
    lines.append(
        err.nlargest(5, "fpr")[["model","task","dataset_number","fpr","fnr","f1","n_actual_neg"]]
        .to_string(index=False) + "\n"
    )

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report_path = out_dir / "stat_results.txt"
    report_path.write_text("".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Optional Excel workbook
    # ------------------------------------------------------------------
    if args.excel:
        excel_path = out_dir / "stat_results.xlsx"
        # Pivot bootstrap so each metric is a column rather than a row
        boot_wide = boot.pivot_table(
            index=["model", "task", "dataset_number", "n"],
            columns="metric",
            values=["point", "ci_low", "ci_high"],
        )
        boot_wide.columns = [f"{m}_{stat}" for stat, m in boot_wide.columns]
        boot_wide = boot_wide.reset_index()
        # Interleave point/ci_low/ci_high per metric for readability
        metric_order = ["accuracy", "f1", "precision", "recall", "mcc"]
        ordered_cols = ["model", "task", "dataset_number", "n"]
        for m in metric_order:
            for stat in ("point", "ci_low", "ci_high"):
                col = f"{m}_{stat}"
                if col in boot_wide.columns:
                    ordered_cols.append(col)
        boot_wide = boot_wide[[c for c in ordered_cols if c in boot_wide.columns]]

        sheets = {
            "class_imbalance":  imb,
            "mcnemar_model":    mc_model,
            "mcnemar_task":     mc_task,
            "mcnemar_overall":  mc_overall,
            "bootstrap_ci":     boot_wide,
            "error_analysis":   err,
        }
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, df_sheet in sheets.items():
                if df_sheet is not None and not df_sheet.empty:
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                    # Auto-fit column widths
                    ws = writer.sheets[sheet_name]
                    for col in ws.columns:
                        max_len = max(
                            (len(str(cell.value)) if cell.value is not None else 0)
                            for cell in col
                        )
                        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)
        print(f"Excel:  {excel_path}")

    print(f"\nReport: {report_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Overall positive rate:   {total_pos/total_n:.3f}")
    if not mc_model.empty:
        print(f"  McNemar model vs model:  {mc_model['significant'].sum()}/{len(mc_model)} significant (raw)")
    if not mc_task.empty:
        print(f"  McNemar task vs task:    {mc_task['significant'].sum()}/{len(mc_task)} significant (raw)")
    if not mc_overall.empty:
        print(f"  McNemar task (overall):  {mc_overall['significant'].sum()}/{len(mc_overall)} significant (raw)")
    print(f"\nOutput files in {out_dir}/:")
    for f in sorted(out_dir.glob("*.*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
