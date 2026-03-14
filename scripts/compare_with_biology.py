#!/usr/bin/env python
"""Compare RL agent trajectories with biological albatross GPS data."""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv
from dynamic_soaring.data.albatross_data import (
    load_albatross_gps,
    gps_to_enu,
    interpolate_trajectory,
    filter_soaring_segments,
)
from dynamic_soaring.analysis.bio_comparison import (
    compute_biological_metrics,
    compare_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare with albatross GPS data")
    parser.add_argument("model", type=str, help="Path to trained model")
    parser.add_argument("--gps-data", type=str, required=True,
                        help="Path to albatross GPS CSV")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="figures/biology/")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()

    # Load and process GPS data
    print("Loading GPS data...")
    gps_df = load_albatross_gps(args.gps_data)
    enu = gps_to_enu(gps_df)
    enu_interp = interpolate_trajectory(enu, dt=config.sim.dt)
    segments = filter_soaring_segments(enu_interp)
    print(f"Found {len(segments)} dynamic soaring segments")

    # Collect RL trajectory
    print("Collecting RL agent trajectory...")
    env = DynamicSoaringEnv(config)
    model = PPO.load(args.model)
    obs, _ = env.reset(seed=42)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    traj_rl = np.array(env.trajectory)

    # Biological metrics
    print("\nBiological metrics (RL agent):")
    rl_metrics = compute_biological_metrics(traj_rl, config.sim.dt, config.bird)
    for k, v in rl_metrics.items():
        print(f"  {k}: {v:.3f}")

    # Compare with each soaring segment
    if segments:
        print("\nComparison with GPS segments:")
        for i, seg in enumerate(segments[:5]):  # top 5
            comparison = compare_trajectories(traj_rl, seg, config.sim.dt)
            print(f"\n  Segment {i}:")
            print(f"    DTW 3D: {comparison['dtw_3d']:.1f}")
            print(f"    DTW altitude: {comparison['dtw_altitude']:.1f}")
            for k, v in comparison.get("metric_diffs", {}).items():
                print(f"    {k} diff: {v:.3f}")
    else:
        print("No soaring segments found in GPS data.")


if __name__ == "__main__":
    main()
