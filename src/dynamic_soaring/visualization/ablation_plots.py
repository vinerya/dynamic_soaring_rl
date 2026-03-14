"""Ablation study visualization: learning curves and comparison charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves_comparison(
    results_dir: str,
    conditions: list[str] | None = None,
    metric: str = "rewards",
    save_path: str | None = None,
) -> None:
    """Plot overlaid learning curves with confidence bands.

    Reads per-seed evaluation data from the experiment results directory.
    """
    results_path = Path(results_dir) / "aggregate_results.json"
    if not results_path.exists():
        print(f"No results at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    if conditions is None:
        conditions = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    for i, cond in enumerate(conditions):
        if cond not in results:
            continue
        values = results[cond].get(metric, [])
        if values:
            ax.bar(i, np.mean(values), yerr=np.std(values),
                   color=colors[i], alpha=0.7, capsize=5, label=cond)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Condition Comparison: {metric}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ablation_heatmap(
    results: dict[str, dict[str, list]],
    metrics: list[str] = ("rewards", "durations", "energy_changes"),
    save_path: str | None = None,
) -> None:
    """Heatmap of normalized performance across conditions and metrics."""
    conditions = list(results.keys())
    n_cond = len(conditions)
    n_met = len(metrics)

    data = np.zeros((n_cond, n_met))
    for i, cond in enumerate(conditions):
        for j, met in enumerate(metrics):
            vals = results[cond].get(met, [0])
            data[i, j] = np.mean(vals)

    # Normalize per metric
    for j in range(n_met):
        col = data[:, j]
        r = col.max() - col.min()
        if r > 0:
            data[:, j] = (col - col.min()) / r

    fig, ax = plt.subplots(figsize=(8, max(4, n_cond * 0.5)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_met))
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_yticks(range(n_cond))
    ax.set_yticklabels(conditions)

    for i in range(n_cond):
        for j in range(n_met):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Normalized Score")
    ax.set_title("Ablation Study: Normalized Performance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
