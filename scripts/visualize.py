#!/usr/bin/env python3
"""CLI entry point for visualization."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv


def main():
    parser = argparse.ArgumentParser(description="Visualize dynamic soaring")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .zip")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument(
        "--mode", type=str, default="trajectory",
        choices=["trajectory", "wind", "training"],
        help="Visualization mode",
    )
    parser.add_argument("--color-by", type=str, default="airspeed",
                        choices=["airspeed", "altitude", "energy"])
    parser.add_argument("--backend", type=str, default="pyvista",
                        choices=["pyvista", "matplotlib"])
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.mode == "wind":
        from dynamic_soaring.visualization.wind_field import plot_wind_profile
        plot_wind_profile(config, save_path=args.save)

    elif args.mode == "training":
        from dynamic_soaring.visualization.training_curves import plot_training_curves
        plot_training_curves(config.training.log_dir, save_path=args.save)

    elif args.mode == "trajectory":
        # Run an episode to get trajectory
        env = DynamicSoaringEnv(config)
        obs, _ = env.reset(seed=args.seed)

        if args.model:
            from stable_baselines3 import PPO
            model = PPO.load(args.model)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        else:
            # Random policy for testing
            print("No model specified, running random policy for visualization test.")
            for _ in range(500):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        trajectory = np.array(env.trajectory)
        print(f"Trajectory: {len(trajectory)} steps, "
              f"alt range [{trajectory[:, 2].min():.1f}, {trajectory[:, 2].max():.1f}]m")

        if args.backend == "pyvista":
            from dynamic_soaring.visualization.trajectory_3d import plot_trajectory_3d
            plot_trajectory_3d(trajectory, config, color_by=args.color_by)
        else:
            from dynamic_soaring.visualization.trajectory_3d import plot_trajectory_matplotlib
            plot_trajectory_matplotlib(trajectory, config, save_path=args.save)


if __name__ == "__main__":
    main()
