import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = "stat_results.xlsx"
DEFAULT_SHEET  = "evaluation"
DEFAULT_METRIC = "f1_score"
DEFAULT_OUTPUT_DIR = "charts"
PALETTE = "mako"

DATASET_ABBREV = {
    "dataset_1": "FZ",
    "dataset_2": "AB",
    "dataset_3": "AG",
    "dataset_4": "DA",
    "dataset_5": "IMTM",
    "dataset_6": "IMTV",
    "dataset_7": "TMTV",
    "dataset_8": "WA",
    "dataset_9": "DS",
}
DATASET_ORDER = ["FZ", "AB", "AG", "DA", "IMTM", "IMTV", "TMTV", "WA", "DS"]

TASK_LABELS = {
    "CandidateSelection_D1Target": "Selection_D1",
    "CandidateSelection_D2Target": "Selection_D2",
    "DualPairs": "DualPairs",
    "Pairs": "Pairs",
}
TASK_ORDER = ["Pairs", "DualPairs", "CandidateSelection_D1Target", "CandidateSelection_D2Target"]

# ---------------------------------------------------------------------------

def plot_model(df: pd.DataFrame, model: str, metric: str, output_dir: Path) -> None:
    model_df = df[df["model"] == model].copy()
    model_df["dataset"] = model_df["dataset"].map(lambda x: DATASET_ABBREV.get(x, x))

    # Order datasets and tasks
    ds_present = [d for d in DATASET_ORDER if d in model_df["dataset"].unique()]
    tasks_present = [t for t in TASK_ORDER if t in model_df["task"].unique()]

    n_tasks = len(tasks_present)
    bar_w   = 0.5
    spacing = n_tasks * bar_w + 0.8
    x       = np.arange(len(ds_present)) * spacing
    colors  = sns.color_palette(PALETTE, n_colors=n_tasks)

    _, ax = plt.subplots(figsize=(14, 6))

    for i, task in enumerate(tasks_present):
        subset = model_df[model_df["task"] == task].set_index("dataset")
        vals   = [subset.loc[ds, metric] if ds in subset.index else 0 for ds in ds_present]
        bars   = ax.bar(x + i * bar_w, vals, bar_w,
                        label=TASK_LABELS.get(task, task),
                        color=colors[i], zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7.5)

    ax.set_xticks(x + n_tasks * bar_w / 2)
    ax.set_xticklabels(ds_present, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, min(ax.get_ylim()[1] * 1.12, 1.15))
    ax.set_title(f"{model} — {metric} by Task and Dataset")
    ax.legend(title="Task")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = output_dir / f"{model}_{metric}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation metrics from stat_results.xlsx")
    parser.add_argument("-i", "--input",  default=DEFAULT_INPUT)
    parser.add_argument("-s", "--sheet",  default=DEFAULT_SHEET)
    parser.add_argument("-m", "--metric", default=DEFAULT_METRIC,
                        help=f"Column to plot (default: {DEFAULT_METRIC})")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, sheet_name=args.sheet)
    models = df["model"].unique()
    print(f"Found {len(models)} models, plotting '{args.metric}'")

    for model in models:
        plot_model(df, model, args.metric, output_dir)


if __name__ == "__main__":
    main()
