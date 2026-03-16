import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric_bars(df: pd.DataFrame, metric: str, palette: str = "tab10", title: str = None) -> None:
    """Plot a grouped bar chart with one bar per task, grouped by dataset."""
    tasks    = df["task"].unique()
    datasets = df["dataset"].unique()
    n_tasks  = len(tasks)
    bar_w    = 0.5
    spacing  = 2.2
    x        = np.arange(len(datasets)) * spacing
    colors   = sns.color_palette(palette, n_colors=n_tasks)

    fig, ax = plt.subplots(figsize=(12, 6))

    task_labels = {
        "CandidateSelection_D1Target": "Selection_D1",
        "CandidateSelection_D2Target": "Selection_D2",
    }

    for i, task in enumerate(tasks):
        subset = df[df["task"] == task].set_index("dataset")
        vals   = [subset.loc[ds, metric] if ds in subset.index else 0 for ds in datasets]
        bars   = ax.bar(x + i * bar_w, vals, bar_w, label=task_labels.get(task, task), color=colors[i], zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5)

    dataset_labels = {
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
    tick_labels = [dataset_labels.get(ds, ds) for ds in datasets]
    ax.set_xticks(x + n_tasks * bar_w / 2)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title if title else f"{metric} by Task and Dataset")
    ax.legend(title="Task")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{metric}_bar_chart.png", dpi=150)
    plt.show()


def main():
    CSV_PATH = "qwen_batch_eval_meta_2302.csv"
    METRIC   = "f1_score" 
    PALETTE  = "mako"

    df = pd.read_csv(CSV_PATH)
    plot_metric_bars(df, metric=METRIC, palette=PALETTE, title=f"Qwen F1 Score by Task and Dataset" )

    CSV_PATH = "llama_batch_eval_meta_2302.csv"
    METRIC   = "f1_score"
    PALETTE  = "mako"

    df = pd.read_csv(CSV_PATH)
    df = df[df["dataset"] != "dataset_3"]
    plot_metric_bars(df, metric=METRIC, palette=PALETTE, title=f"Llama F1 Score by Task and Dataset" )


if __name__ == "__main__":
    main()