#!/usr/bin/env python3
"""CLI entry point for evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_soaring.config import Config
from dynamic_soaring.evaluation.evaluate import evaluate_policy, print_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained dynamic soaring agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    results = evaluate_policy(
        model_path=args.model,
        config=config,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    print_evaluation_report(results)


if __name__ == "__main__":
    main()
