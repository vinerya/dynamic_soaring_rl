#!/usr/bin/env python3
"""CLI entry point for training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_soaring.config import Config
from dynamic_soaring.training.train import train


def main():
    parser = argparse.ArgumentParser(description="Train dynamic soaring agent with PPO")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.timesteps is not None:
        config.training.total_timesteps = args.timesteps
    if args.seed is not None:
        config.training.seed = args.seed

    print(f"Starting training: {config.training.total_timesteps} timesteps, "
          f"seed={config.training.seed}, algorithm={config.training.algorithm}")
    print(f"Wind: {config.wind.profile_type}, U_ref={config.wind.reference_speed} m/s")

    model = train(config)
    print(f"\nTraining complete. Model saved to {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
