#!/usr/bin/env python
"""Run generalization experiments: transfer evaluation and robustness curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_soaring.config import Config
from dynamic_soaring.evaluation.transfer_eval import evaluate_transfer, compute_robustness_curve
from dynamic_soaring.visualization.generalization_plots import (
    plot_transfer_matrix,
    plot_robustness_curve,
    plot_specialist_vs_generalist,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generalization evaluation")
    parser.add_argument("model", type=str, help="Path to trained model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="figures/generalization/")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--specialist-models", type=str, nargs="*", default=None,
                        help="Paths to specialist models for comparison")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Transfer evaluation across wind speeds
    print("Running transfer evaluation...")
    wind_configs = {}
    for speed in [5, 8, 10, 12, 15, 18, 20, 25]:
        cfg = config.merge_overrides({"wind": {"reference_speed": float(speed)}})
        wind_configs[f"{speed}m/s"] = cfg

    transfer_results = evaluate_transfer(
        args.model, wind_configs, n_episodes=args.n_episodes,
    )
    print("\nTransfer Results:")
    for name, r in transfer_results.items():
        print(f"  {name}: reward={r['reward_mean']:.1f} ± {r['reward_std']:.1f}")

    plot_transfer_matrix(transfer_results, save_path=str(save_dir / "transfer_matrix.png"))

    # Robustness curve
    print("\nComputing robustness curve...")
    robustness = compute_robustness_curve(
        args.model, config,
        param_name="reference_speed",
        param_values=[5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0],
        n_episodes=args.n_episodes,
    )
    plot_robustness_curve(
        robustness, "wind_speed",
        save_path=str(save_dir / "robustness_wind_speed.png"),
    )

    # Save results
    results_path = save_dir / "generalization_results.json"
    with open(results_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
                   for k, v in transfer_results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
