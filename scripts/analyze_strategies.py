#!/usr/bin/env python
"""Analyze learned soaring strategies: phase portraits, classification, energy."""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv
from dynamic_soaring.analysis.rayleigh_cycle import generate_rayleigh_cycle
from dynamic_soaring.analysis.trajectory_classifier import classify_trajectory_pattern
from dynamic_soaring.analysis.phase_space import compute_phase_portraits, compare_phase_portraits
from dynamic_soaring.analysis.energy_extraction import (
    compute_energy_extraction_rate,
    compute_energy_efficiency,
    decompose_energy_budget,
)
from dynamic_soaring.visualization.phase_plots import (
    plot_phase_portrait,
    plot_trajectory_comparison,
    plot_energy_budget,
)


def collect_trajectory(model_path: str, config: Config, seed: int = 42) -> np.ndarray:
    """Run one episode and return trajectory."""
    env = DynamicSoaringEnv(config)
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.array(env.trajectory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze soaring strategies")
    parser.add_argument("model", type=str, help="Path to trained model")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--save-dir", type=str, default="figures/", help="Save directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    save_dir = args.save_dir

    # Collect agent trajectory
    print("Collecting agent trajectory...")
    traj_agent = collect_trajectory(args.model, config, args.seed)

    # Generate Rayleigh baseline
    print("Generating Rayleigh cycle baseline...")
    traj_baseline = generate_rayleigh_cycle(config.wind, config.bird)

    # Classify pattern
    pattern = classify_trajectory_pattern(traj_agent, config.sim.dt)
    print(f"\nTrajectory Pattern: {pattern['pattern']} (confidence: {pattern['confidence']:.2f})")
    print(f"  Dominant period: {pattern['dominant_period']:.1f}s")
    print(f"  Heading rate: {pattern['heading_rate_mean']:.2f} rad/s")

    # Phase portraits
    print("\nComputing phase portraits...")
    portraits_agent = compute_phase_portraits(traj_agent, config.sim.dt)
    portraits_baseline = compute_phase_portraits(traj_baseline, config.sim.dt)

    comparison = compare_phase_portraits(portraits_agent, portraits_baseline)
    print("Phase portrait similarity (lower = more similar):")
    for name, score in comparison.items():
        print(f"  {name}: {score:.3f}")

    # Energy analysis
    print("\nEnergy analysis...")
    extraction = compute_energy_extraction_rate(traj_agent, config.sim.dt)
    efficiency = compute_energy_efficiency(traj_agent, config.sim.dt, config.bird)
    budget = decompose_energy_budget(traj_agent, config.sim.dt, config.bird, config.wind)

    print(f"  Mean extraction rate: {np.mean(extraction):.2f} J/(kg·s)")
    print(f"  Energy efficiency: {efficiency:.3f}")
    print(f"  Wind extraction: {budget['wind_extraction']:.1f} J/kg")
    print(f"  Drag loss: {budget['drag_loss']:.1f} J/kg")

    # Plots
    print("\nGenerating plots...")
    plot_trajectory_comparison(traj_agent, traj_baseline, save_path=f"{save_dir}/strategy_comparison.png")
    plot_energy_budget(budget, save_path=f"{save_dir}/energy_budget.png")
    plot_phase_portrait(
        portraits_agent["altitude_airspeed"],
        "Altitude (m)", "Airspeed (m/s)", "Altitude-Airspeed Phase Portrait",
        save_path=f"{save_dir}/phase_alt_airspeed.png",
    )

    print("Done!")


if __name__ == "__main__":
    main()
