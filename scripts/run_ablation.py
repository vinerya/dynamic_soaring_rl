#!/usr/bin/env python
"""Run reward/curriculum ablation and generate comparison plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_soaring.experiments.experiment_config import load_experiment
from dynamic_soaring.experiments.runner import run_experiment
from dynamic_soaring.experiments.analysis import compare_conditions, generate_latex_table
from dynamic_soaring.visualization.ablation_plots import (
    plot_learning_curves_comparison,
    plot_ablation_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("spec", type=str, help="Path to ablation experiment YAML")
    parser.add_argument("--output-dir", type=str, default="results/ablation/")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, only plot existing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not args.plot_only:
        spec = load_experiment(args.spec)
        run_experiment(spec, output_dir=str(output_dir))

    # Load and analyze results
    results_file = output_dir / "aggregate_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        # Statistical comparison
        comparison = compare_conditions(results)
        print("\n" + "=" * 60)
        print("STATISTICAL COMPARISON")
        print("=" * 60)
        for pair, stats in comparison.items():
            print(f"\n{pair}:")
            print(f"  U-statistic: {stats.get('u_statistic', 'N/A')}")
            print(f"  p-value: {stats.get('p_value', 'N/A')}")
            print(f"  Significant: {stats.get('significant', 'N/A')}")

        # LaTeX table
        latex = generate_latex_table(results)
        latex_path = output_dir / "results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to {latex_path}")

        # Plots
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        plot_learning_curves_comparison(
            str(output_dir),
            save_path=str(figures_dir / "learning_curves.png"),
        )
        plot_ablation_heatmap(
            results,
            save_path=str(figures_dir / "ablation_heatmap.png"),
        )
    else:
        print(f"No results found at {results_file}")


if __name__ == "__main__":
    main()
