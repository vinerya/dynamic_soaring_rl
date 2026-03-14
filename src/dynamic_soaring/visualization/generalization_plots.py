"""Generalization and transfer evaluation visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_transfer_matrix(
    results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Heatmap of model performance across wind conditions.

    Args:
        results: dict mapping condition_name -> {reward_mean, ...}
    """
    conditions = list(results.keys())
    n = len(conditions)
    rewards = [results[c]["reward_mean"] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(n), rewards, color="steelblue", alpha=0.8, edgecolor="black")

    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(n))
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Transfer Performance Across Wind Conditions")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_robustness_curve(
    data: dict[str, np.ndarray],
    param_name: str,
    models: dict[str, dict] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot performance vs parameter sweep with confidence bands.

    Args:
        data: dict with param_values, reward_means, reward_stds
        param_name: name of the swept parameter
        models: optional dict of additional model curves to overlay
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data["param_values"]
    y = data["reward_means"]
    err = data["reward_stds"]

    ax.plot(x, y, "b-o", linewidth=2, markersize=4, label="Model")
    ax.fill_between(x, y - err, y + err, alpha=0.2, color="blue")

    if models:
        colors = ["red", "green", "orange", "purple"]
        for i, (name, d) in enumerate(models.items()):
            c = colors[i % len(colors)]
            ax.plot(d["param_values"], d["reward_means"], f"-o",
                    color=c, linewidth=2, markersize=4, label=name)
            ax.fill_between(d["param_values"],
                            d["reward_means"] - d["reward_stds"],
                            d["reward_means"] + d["reward_stds"],
                            alpha=0.15, color=c)

    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Robustness: Performance vs {param_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_specialist_vs_generalist(
    specialist_results: dict[str, dict],
    generalist_results: dict[str, dict],
    context_results: dict[str, dict] | None = None,
    save_path: str | None = None,
) -> None:
    """Grouped bar chart comparing specialist, generalist, and context-conditioned."""
    conditions = list(specialist_results.keys())
    n = len(conditions)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    spec_vals = [specialist_results[c]["reward_mean"] for c in conditions]
    gen_vals = [generalist_results[c]["reward_mean"] for c in conditions]

    ax.bar(x - width, spec_vals, width, label="Specialist", color="steelblue", alpha=0.8)
    ax.bar(x, gen_vals, width, label="Generalist", color="coral", alpha=0.8)

    if context_results:
        ctx_vals = [context_results[c]["reward_mean"] for c in conditions]
        ax.bar(x + width, ctx_vals, width, label="Context-Conditioned", color="seagreen", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Specialist vs Generalist Performance")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
