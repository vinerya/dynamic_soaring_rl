#!/usr/bin/env python
"""Run an ablation or comparison experiment from a YAML spec."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_soaring.experiments.experiment_config import load_experiment
from dynamic_soaring.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dynamic soaring experiment")
    parser.add_argument("spec", type=str, help="Path to experiment YAML spec")
    parser.add_argument("--output-dir", type=str, default="results/",
                        help="Output directory for results")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds from spec")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Run only these conditions (default: all)")
    args = parser.parse_args()

    spec = load_experiment(args.spec)
    if args.seeds:
        spec.seeds = args.seeds
    if args.conditions:
        spec.conditions = [c for c in spec.conditions if c.name in args.conditions]

    run_experiment(spec, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
