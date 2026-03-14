"""Statistical analysis for experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_experiment_results(results_dir: str | Path) -> dict[str, dict[str, list]]:
    """Load aggregate results from an experiment directory."""
    path = Path(results_dir) / "aggregate_results.json"
    with open(path) as f:
        return json.load(f)


def compare_conditions(
    results: dict[str, dict[str, list]],
    metric: str = "rewards",
    baseline: str | None = None,
) -> dict:
    """Compare all conditions on a metric with statistical tests.

    Returns dict with per-condition stats and pairwise comparisons to baseline.
    """
    summary = {}
    for name, data in results.items():
        values = np.array(data[metric])
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": len(values),
        }

    # Pairwise tests against baseline
    if baseline and baseline in results:
        baseline_vals = np.array(results[baseline][metric])
        for name, data in results.items():
            if name == baseline:
                summary[name]["p_value"] = 1.0
                continue
            test_vals = np.array(data[metric])
            if len(baseline_vals) >= 3 and len(test_vals) >= 3:
                stat, p = stats.mannwhitneyu(baseline_vals, test_vals, alternative="two-sided")
                summary[name]["p_value"] = float(p)
                summary[name]["effect_size"] = float(np.mean(test_vals) - np.mean(baseline_vals))
            else:
                summary[name]["p_value"] = None

    return summary


def generate_latex_table(
    comparison: dict,
    metric_name: str = "Reward",
    caption: str = "Ablation results",
) -> str:
    """Generate a LaTeX table from comparison results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        f"Condition & {metric_name} (mean) & Std & p-value & Effect \\\\",
        r"\hline",
    ]
    for name, data in comparison.items():
        p = data.get("p_value")
        p_str = f"{p:.4f}" if p is not None else "--"
        effect = data.get("effect_size")
        e_str = f"{effect:+.3f}" if effect is not None else "--"
        lines.append(f"{name} & {data['mean']:.3f} & {data['std']:.3f} & {p_str} & {e_str} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
