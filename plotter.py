import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric_bars(df: pd.DataFrame, metric: str, palette: str = "tab10", title: str = None) -> None:
    """Plot a grouped bar chart with one bar per task, grouped by dataset."""
    tasks    = df["task"].unique()
    datasets = df["dataset"].unique()
    bar_w    = 0.25
    x        = np.arange(len(datasets))
    colors   = sns.color_palette(palette, n_colors=len(tasks))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, task in enumerate(tasks):
        subset = df[df["task"] == task].set_index("dataset")
        vals   = [subset.loc[ds, metric] if ds in subset.index else 0 for ds in datasets]
        bars   = ax.bar(x + i * bar_w, vals, bar_w, label=task, color=colors[i], zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5)

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title if title else f"{metric} by Task and Dataset")
    ax.legend(title="Task")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{metric}_bar_chart.png", dpi=150)
    plt.show()


def main():
    CSV_PATH = "batch_results/qwen_batch_eval_meta_no3.csv"
    METRIC   = "f1_score" 
    PALETTE  = "mako"

    df = pd.read_csv(CSV_PATH)
    plot_metric_bars(df, metric=METRIC, palette=PALETTE, title=f"Qwen F1 Score by Task and Dataset" )


if __name__ == "__main__":
    main()