"""
Generate LaTeX appendix tables from stat_results.xlsx.

Outputs (all written to --output-dir, default: latex_tables/appendix/):
  class_imbalance_<model>.tex
  bootstrap_ci_<model>.tex
  mcnemar_overall_<model>.tex
  mcnemar_model_<model>_<dataset>.tex   (per dataset)
  mcnemar_task_<model>_<dataset>.tex    (per dataset)

Usage:
    python appendix_tables.py
    python appendix_tables.py -i analysis_output/stat_results.xlsx -o latex_tables/appendix
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_INPUT      = "stat_results.xlsx"
DEFAULT_OUTPUT_DIR = "latex_tables/appendix1"

DATASET_ABBREV = {
    1: "FZ", 2: "AB", 3: "AG", 4: "DA", 5: "IMTM",
    6: "IMTV", 7: "TMTV", 8: "WA", 9: "DS",
}
DATASET_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9]

TASK_LABELS = {
    "CandidateSelection_D1Target": "Sel\\_D1",
    "CandidateSelection_D2Target": "Sel\\_D2",
    "DualPairs": "DualPairs",
    "Pairs": "Pairs",
}
TASK_ORDER = ["Pairs", "DualPairs", "CandidateSelection_D1Target", "CandidateSelection_D2Target"]

ROUND = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r(val, digits=ROUND) -> str:
    try:
        f = float(val)
        return str(round(f, digits))
    except (TypeError, ValueError):
        return str(val)


def fmt_p(val) -> str:
    """Format a p-value: scientific notation if < 0.001, otherwise 4 decimal places."""
    try:
        f = float(val)
        if f == 0.0:
            return "$< 10^{-15}$"
        if f < 0.001:
            s = f"{f:.2e}"
            mantissa, exp = s.split("e")
            exp_int = int(exp)
            return f"${mantissa} \\times 10^{{{exp_int}}}$"
        return str(round(f, 4))
    except (TypeError, ValueError):
        return str(val)


def sig_marker(sig: bool) -> str:
    return r"\textbf{*}" if sig else ""


def ds_label(num) -> str:
    return DATASET_ABBREV.get(int(num), str(num))


def task_label(name: str) -> str:
    return TASK_LABELS.get(name, name)


def model_label(name: str) -> str:
    return name.replace("_", "\\_")


def write_tex(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  {path}")


def wrap_table(body: str, caption: str, label: str, col_spec: str,
               wide: bool = False) -> str:
    env = "table*" if wide else "table"
    return "\n".join([
        f"\\begin{{{env}}}[]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        body,
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\end{{{env}}}",
    ])


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------

def make_class_imbalance(df: pd.DataFrame, model: str, output_dir: Path) -> None:
    mdf = df[df["model"] == model].copy()

    # Sort by dataset then task
    ds_order   = {v: i for i, v in enumerate(DATASET_ORDER)}
    task_order = {v: i for i, v in enumerate(TASK_ORDER)}
    mdf["_ds"]   = mdf["dataset_number"].map(lambda x: ds_order.get(int(x), 99))
    mdf["_task"] = mdf["task"].map(lambda x: task_order.get(x, 99))
    mdf = mdf.sort_values(["_ds", "_task"]).reset_index(drop=True)

    header = (
        r"\multicolumn{1}{|l|}{\textbf{dataset}} & "
        r"\multicolumn{1}{l|}{\textbf{task}} & "
        r"\multicolumn{1}{l|}{\textbf{pos rate}} & "
        r"\multicolumn{1}{l|}{\textbf{imb ratio}} & "
        r"\multicolumn{1}{l|}{\textbf{F1}} & "
        r"\multicolumn{1}{l|}{\textbf{MCC}} \\ \hline"
    )

    rows = []
    for ds_num in sorted(mdf["dataset_number"].unique(),
                         key=lambda x: ds_order.get(int(x), 99)):
        ds_rows = mdf[mdf["dataset_number"] == ds_num]
        n = len(ds_rows)
        ds = ds_label(ds_num)
        for i, (_, row) in enumerate(ds_rows.iterrows()):
            ds_cell = f"\\multirow{{{n}}}{{*}}{{{ds}}}" if i == 0 else ""
            rows.append(
                f"{ds_cell} & {task_label(row['task'])} & "
                f"{r(row['pos_rate'])} & {r(row['imb_ratio'])} & "
                f"{r(row['f1'])} & {r(row['mcc'])} \\\\"
            )
        rows[-1] += r" \hline"

    body = header + "\n" + "\n".join(rows)
    caption = f"Class imbalance — {model_label(model)}"
    label   = f"tab:imbalance_{model}"
    tex = wrap_table(body, caption, label, "|l|l|l|l|l|l|", wide=True)

    write_tex(output_dir / f"class_imbalance_{model}.tex", tex)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def make_bootstrap_ci(df: pd.DataFrame, model: str, output_dir: Path) -> None:
    mdf = df[df["model"] == model].copy()

    ds_order   = {v: i for i, v in enumerate(DATASET_ORDER)}
    task_order = {v: i for i, v in enumerate(TASK_ORDER)}
    mdf["_ds"]   = mdf["dataset_number"].map(lambda x: ds_order.get(int(x), 99))
    mdf["_task"] = mdf["task"].map(lambda x: task_order.get(x, 99))
    mdf = mdf.sort_values(["_ds", "_task"]).reset_index(drop=True)

    header = (
        r"\multicolumn{1}{|l|}{\textbf{dataset}} & "
        r"\multicolumn{1}{l|}{\textbf{task}} & "
        r"\multicolumn{1}{l|}{\textbf{F1}} & "
        r"\multicolumn{1}{l|}{\textbf{F1 CI low}} & "
        r"\multicolumn{1}{l|}{\textbf{F1 CI high}} & "
        r"\multicolumn{1}{l|}{\textbf{MCC}} & "
        r"\multicolumn{1}{l|}{\textbf{MCC CI low}} & "
        r"\multicolumn{1}{l|}{\textbf{MCC CI high}} \\ \hline"
    )

    rows = []
    for ds_num in sorted(mdf["dataset_number"].unique(),
                         key=lambda x: ds_order.get(int(x), 99)):
        ds_rows = mdf[mdf["dataset_number"] == ds_num]
        n = len(ds_rows)
        ds = ds_label(ds_num)
        for i, (_, row) in enumerate(ds_rows.iterrows()):
            ds_cell = f"\\multirow{{{n}}}{{*}}{{{ds}}}" if i == 0 else ""
            rows.append(
                f"{ds_cell} & {task_label(row['task'])} & "
                f"{r(row['f1_point'])} & {r(row['f1_ci_low'])} & {r(row['f1_ci_high'])} & "
                f"{r(row['mcc_point'])} & {r(row['mcc_ci_low'])} & {r(row['mcc_ci_high'])} \\\\"
            )
        rows[-1] += r" \hline"

    body = header + "\n" + "\n".join(rows)
    caption = f"Bootstrap 95\\% CI — {model_label(model)}"
    label   = f"tab:bootstrap_{model}"
    tex = wrap_table(body, caption, label, "|l|l|l|l|l|l|l|l|", wide=True)

    write_tex(output_dir / f"bootstrap_ci_{model}.tex", tex)


# ---------------------------------------------------------------------------
# McNemar overall (task vs task, aggregated across datasets, per model)
# ---------------------------------------------------------------------------

def make_mcnemar_overall(df: pd.DataFrame, model: str, output_dir: Path) -> None:
    mdf = df[df["model"] == model].copy()

    header = (
        r"\multicolumn{1}{|l|}{\textbf{task A}} & "
        r"\multicolumn{1}{l|}{\textbf{task B}} & "
        r"\multicolumn{1}{l|}{\textbf{acc A}} & "
        r"\multicolumn{1}{l|}{\textbf{acc B}} & "
        r"\multicolumn{1}{l|}{\textbf{p (Bonf.)}} & "
        r"\multicolumn{1}{l|}{\textbf{p (raw)}} & "
        r"\multicolumn{1}{l|}{\textbf{sig.}} \\ \hline"
    )

    rows = []
    for _, row in mdf.iterrows():
        sig = sig_marker(bool(row["sig_bonferroni"]))
        rows.append(
            f"{task_label(row['name_a'])} & {task_label(row['name_b'])} & "
            f"{r(row['rate_a'])} & {r(row['rate_b'])} & "
            f"{fmt_p(row['p_bonferroni'])} & {fmt_p(row['p_value'])} & {sig} \\\\ \\hline"
        )

    body = header + "\n" + "\n".join(rows)
    caption = f"McNemar overall (task comparison) — {model_label(model)}"
    label   = f"tab:mcnemar_overall_{model}"
    tex = wrap_table(body, caption, label, "|l|l|l|l|l|l|l|")

    write_tex(output_dir / f"mcnemar_overall_{model}.tex", tex)


# ---------------------------------------------------------------------------
# McNemar model (model vs model, per dataset per task)
# ---------------------------------------------------------------------------

def make_mcnemar_model(df: pd.DataFrame, output_dir: Path) -> None:
    ds_order   = {v: i for i, v in enumerate(DATASET_ORDER)}
    task_order = {v: i for i, v in enumerate(TASK_ORDER)}

    for ds_num in sorted(df["dataset_number"].unique(),
                         key=lambda x: ds_order.get(int(x), 99)):
        ds = ds_label(ds_num)
        ddf = df[df["dataset_number"] == ds_num].copy()
        ddf["_task"] = ddf["task"].map(lambda x: task_order.get(x, 99))
        ddf = ddf.sort_values("_task").reset_index(drop=True)

        header = (
            r"\textbf{Task} & \textbf{Model A} & \textbf{Model B} & "
            r"\textbf{Acc A} & \textbf{Acc B} & "
            r"\textbf{p (Bonf.)} & \textbf{Sig.} \\ \hline"
        )

        rows = []
        prev_task = None
        for _, row in ddf.iterrows():
            tl = task_label(row["task"])
            task_cell = tl if tl != prev_task else ""
            prev_task = tl
            sig = sig_marker(bool(row["sig_bonferroni"]))
            rows.append(
                f"{task_cell} & {model_label(row['name_a'])} & {model_label(row['name_b'])} & "
                f"{r(row['rate_a'])} & {r(row['rate_b'])} & "
                f"{fmt_p(row['p_bonferroni'])} & {sig} \\\\"
            )

        body = header + "\n" + "\n".join(rows)
        caption = f"McNemar model comparison — dataset {ds}"
        label   = f"tab:mcnemar_model_{ds}"
        tex = wrap_table(body, caption, label, "|l|l|l|r|r|r|c|", wide=True)

        write_tex(output_dir / f"mcnemar_model_{ds}.tex", tex)


# ---------------------------------------------------------------------------
# McNemar task (task vs task, per dataset per model)
# ---------------------------------------------------------------------------

def make_mcnemar_task(df: pd.DataFrame, output_dir: Path) -> None:
    ds_order   = {v: i for i, v in enumerate(DATASET_ORDER)}

    for ds_num in sorted(df["dataset_number"].unique(),
                         key=lambda x: ds_order.get(int(x), 99)):
        ds = ds_label(ds_num)
        ddf = df[df["dataset_number"] == ds_num].copy()

        header = (
            r"\textbf{Model} & \textbf{Task A} & \textbf{Task B} & "
            r"\textbf{Acc A} & \textbf{Acc B} & "
            r"\textbf{p (Bonf.)} & \textbf{Sig.} \\ \hline"
        )

        rows = []
        prev_model = None
        for _, row in ddf.iterrows():
            ml = model_label(row["model"])
            model_cell = ml if ml != prev_model else ""
            prev_model = ml
            sig = sig_marker(bool(row["sig_bonferroni"]))
            rows.append(
                f"{model_cell} & {task_label(row['name_a'])} & {task_label(row['name_b'])} & "
                f"{r(row['rate_a'])} & {r(row['rate_b'])} & "
                f"{fmt_p(row['p_bonferroni'])} & {sig} \\\\"
            )

        body = header + "\n" + "\n".join(rows)
        caption = f"McNemar task comparison — dataset {ds}"
        label   = f"tab:mcnemar_task_{ds}"
        tex = wrap_table(body, caption, label, "|l|l|l|r|r|r|c|", wide=True)

        write_tex(output_dir / f"mcnemar_task_{ds}.tex", tex)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def make_error_analysis(df: pd.DataFrame, model: str, output_dir: Path) -> None:
    mdf = df[df["model"] == model].copy()

    ds_order   = {v: i for i, v in enumerate(DATASET_ORDER)}
    task_order = {v: i for i, v in enumerate(TASK_ORDER)}
    mdf["_ds"]   = mdf["dataset_number"].map(lambda x: ds_order.get(int(x), 99))
    mdf["_task"] = mdf["task"].map(lambda x: task_order.get(x, 99))
    mdf = mdf.sort_values(["_ds", "_task"]).reset_index(drop=True)

    header = (
        r"\multicolumn{1}{|l|}{\textbf{dataset}} & "
        r"\multicolumn{1}{l|}{\textbf{task}} & "
        r"\multicolumn{1}{l|}{\textbf{TP}} & "
        r"\multicolumn{1}{l|}{\textbf{FP}} & "
        r"\multicolumn{1}{l|}{\textbf{FN}} & "
        r"\multicolumn{1}{l|}{\textbf{TN}} & "
        r"\multicolumn{1}{l|}{\textbf{FPR}} & "
        r"\multicolumn{1}{l|}{\textbf{FNR}} & "
        r"\multicolumn{1}{l|}{\textbf{error type}} \\ \hline"
    )

    rows = []
    for ds_num in sorted(mdf["dataset_number"].unique(),
                         key=lambda x: ds_order.get(int(x), 99)):
        ds_rows = mdf[mdf["dataset_number"] == ds_num]
        n = len(ds_rows)
        ds = ds_label(ds_num)
        for i, (_, row) in enumerate(ds_rows.iterrows()):
            ds_cell = f"\\multirow{{{n}}}{{*}}{{{ds}}}" if i == 0 else ""
            rows.append(
                f"{ds_cell} & {task_label(row['task'])} & "
                f"{int(row['tp'])} & {int(row['fp'])} & "
                f"{int(row['fn'])} & {int(row['tn'])} & "
                f"{r(row['fpr'])} & {r(row['fnr'])} & "
                f"{row['error_type']} \\\\"
            )
        rows[-1] += r" \hline"

    body = header + "\n" + "\n".join(rows)
    caption = f"Error analysis — {model_label(model)}"
    label   = f"tab:error_{model}"
    tex = wrap_table(body, caption, label, "|l|l|l|l|l|l|l|l|l|", wide=True)

    write_tex(output_dir / f"error_analysis_{model}.tex", tex)


# ---------------------------------------------------------------------------
# Token consumption
# ---------------------------------------------------------------------------

# Rates per 1M tokens (ChatGPT-5.4-mini)
PRICE_INPUT_PER_M  = 0.75
PRICE_OUTPUT_PER_M = 4.5

# Task order for token table: Sel_D1, Sel_D2, DualPairs, Pairs
TOKEN_TASK_ORDER = [
    "CandidateSelection_D1Target",
    "CandidateSelection_D2Target",
    "DualPairs",
    "Pairs",
]


def make_token_consumption(df: pd.DataFrame, model: str, output_dir: Path) -> None:
    mdf = df[df["model"] == model].copy()

    ds_order   = {v: i for i, v in enumerate(DATASET_ORDER)}
    task_order = {v: i for i, v in enumerate(TOKEN_TASK_ORDER)}
    mdf["_ds"]   = mdf["dataset_number"].map(lambda x: ds_order.get(int(x), 99))
    mdf["_task"] = mdf["task"].map(lambda x: task_order.get(x, 99))
    mdf = mdf.sort_values(["_ds", "_task"]).reset_index(drop=True)

    mdf["price"] = (
        mdf["token_prompt_tokens"]     / 1_000_000 * PRICE_INPUT_PER_M +
        mdf["token_completion_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
    ).round(2)

    # Index of cheapest row per dataset (for bold)
    cheapest = set(mdf.groupby("dataset_number")["price"].idxmin().values)

    header = (
        r"\multicolumn{1}{|l|}{\textbf{dataset}} & "
        r"\multicolumn{1}{l|}{\textbf{task}} & "
        r"\multicolumn{1}{l|}{\textbf{llm calls}} & "
        r"\multicolumn{1}{l|}{\textbf{avg total tokens}} & "
        r"\multicolumn{1}{l|}{\textbf{input tokens}} & "
        r"\multicolumn{1}{l|}{\textbf{output tokens}} & "
        r"\multicolumn{1}{l|}{\textbf{total tokens}} & "
        r"\multicolumn{1}{l|}{\textbf{total price (\euro{})}} \\ \hline"
    )

    rows = []
    for ds_num in sorted(mdf["dataset_number"].unique(),
                         key=lambda x: ds_order.get(int(x), 99)):
        ds_rows = mdf[mdf["dataset_number"] == ds_num]
        n = len(ds_rows)
        ds = ds_label(ds_num)

        for i, (idx, row) in enumerate(ds_rows.iterrows()):
            ds_cell = f"\\multirow{{{n}}}{{*}}{{{ds}}}" if i == 0 else ""
            price_str = r(row["price"], 2)
            if idx in cheapest:
                price_str = r"\textbf{" + price_str + r"}"
            rows.append(
                f"{ds_cell} & {task_label(row['task'])} & "
                f"{int(row['token_llm_calls'])} & "
                f"{round(row['token_avg_total_tokens'], 2)} & "
                f"{int(row['token_prompt_tokens'])} & "
                f"{int(row['token_completion_tokens'])} & "
                f"{int(row['token_total_tokens'])} & "
                f"{price_str} \\\\"
            )
        rows[-1] += r" \hline"

    col_spec = "|l|l|l|l|l|l|l|l|"
    body = header + "\n" + "\n".join(rows)
    caption = (
        "Token consumption and estimated API cost per dataset and prompting strategy "
        "(ChatGPT-5-mini rates)"
    )
    label = f"tab:tokens_{model}"

    tex = "\n".join([
        r"\begin{table*}[ht]",
        r"\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        body,
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ])

    write_tex(output_dir / f"token_consumption_{model}.tex", tex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX appendix tables")
    parser.add_argument("-i", "--input",      default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    output_dir = Path(args.output_dir)
    xl = pd.ExcelFile(input_path)

    print("Class imbalance:")
    ci_df = xl.parse("class_imbalance")
    for model in ci_df["model"].unique():
        make_class_imbalance(ci_df, model, output_dir)

    print("\nBootstrap CI:")
    bs_df = xl.parse("bootstrap_ci")
    for model in bs_df["model"].unique():
        make_bootstrap_ci(bs_df, model, output_dir)

    print("\nMcNemar overall:")
    mo_df = xl.parse("mcnemar_overall")
    for model in mo_df["model"].unique():
        make_mcnemar_overall(mo_df, model, output_dir)

    print("\nError analysis:")
    ea_df = xl.parse("error_analysis")
    for model in ea_df["model"].unique():
        make_error_analysis(ea_df, model, output_dir)

    print("\nToken consumption:")
    ev_df = xl.parse("evaluation")
    for model in ev_df["model"].unique():
        make_token_consumption(ev_df, model, output_dir)


if __name__ == "__main__":
    main()
