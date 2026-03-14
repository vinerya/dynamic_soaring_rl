#!/usr/bin/env python
"""Analyze multi-agent flock behavior from trained models."""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from dynamic_soaring.config import Config
from dynamic_soaring.envs.multi_agent_env import MultiAgentSoaringEnv
from dynamic_soaring.analysis.flock_analysis import (
    compute_flock_metrics,
    compute_cooperation_benefit,
)
from dynamic_soaring.visualization.flock_visualization import (
    plot_multi_agent_trajectory,
    plot_formation_snapshots,
    plot_inter_agent_distances,
)


def collect_multi_agent_trajectories(
    model_path: str, config: Config, seed: int = 42,
) -> list[np.ndarray]:
    """Run one episode and return per-agent trajectories."""
    env = MultiAgentSoaringEnv(config)
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return [np.array(t) for t in env.agent_trajectories]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze flock behavior")
    parser.add_argument("model", type=str, help="Path to trained multi-agent model")
    parser.add_argument("--config", type=str, default="configs/multi_agent.yaml")
    parser.add_argument("--save-dir", type=str, default="figures/flock/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    save_dir = args.save_dir

    print("Collecting multi-agent trajectories...")
    trajectories = collect_multi_agent_trajectories(args.model, config, args.seed)
    print(f"  {len(trajectories)} agents, lengths: {[len(t) for t in trajectories]}")

    # Flock metrics
    metrics = compute_flock_metrics(trajectories, config.sim.dt)
    print("\nFlock Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Plots
    print("\nGenerating plots...")
    plot_multi_agent_trajectory(trajectories, save_path=f"{save_dir}/trajectories_3d.png")
    plot_formation_snapshots(
        trajectories,
        times=[5.0, 15.0, 30.0, 50.0],
        dt=config.sim.dt,
        save_path=f"{save_dir}/formation_snapshots.png",
    )
    plot_inter_agent_distances(trajectories, config.sim.dt, save_path=f"{save_dir}/distances.png")

    print("Done!")


if __name__ == "__main__":
    main()
